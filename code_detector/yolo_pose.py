import time, random, yaml, cv2, os, json
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO 
from ROI import ROIHandler
from tb_gateway_mqtt import TBGatewayMqttClient
from deltalake.writer import write_deltalake
from labeling.labeling_batch import Labeller
from labeling.labeling_utils import process_df

ROIS_CONFIG_PATH = "video_server/rois_url0.yaml"

class StereoDetector:
    def __init__(self, config_paths, roi_path, model_params):

        # --- Load config ---
        with open(config_paths["config"], "r") as file:
            self.config = yaml.safe_load(file)
        with open(roi_path, "r") as file:
            roi_data = yaml.safe_load(file)

        self.model = YOLO(model_params["model_path"]) 
        print("Modelo carregado!")

        # --- Labeller ---
        self.labeller = Labeller(device_id=self.config["device_id"])
        self.saved_count = 0

        # --- STREAMLIT ---
        self.is_running = False
        self.save_interval = 10
        self.max_saves = 10
        self.saved_count = 0

        try:
            self.client_control = self.labeller.mqtt_com.client 
            self.client_control.subscribe("system/control/#")
            self.client_control.message_callback_add("system/control/#", self.on_control_message)
            print("🎧 À ESCUTA EM 'system/control'...")
        except:
            print("⚠️ AVISO: Não consegui ligar ao callback de controlo.")

        # --- Image Configs ---
        self.resolution = (
            self.config["resolution"]["height"],
            self.config["resolution"]["width"]
        )

        # --- Default Payload ---
        self.default_payload = {                            
            "ts": None,
            "roi_id": 1000000,
            "label_type": "none",
            "confidence": -999,
            "bbox_coords": (-500, -500, -500, -500),
            "keypoints": [[-1, -1, -1] for _ in range(17)]  # Lista de 17 keypoints padrão
        }

        # --- Export to parquet --- #
        self.parquet_buffer = []
        self.parquet_raw_path = f"data_exports/raw_{self.config['device_id']}"
        self.parquet_labelled_path = f"data_exports/labelled_{self.config['device_id']}"
        self.last_save_time = time.time()

        # --- Thingsboard ---
        # self.tb_client = self.config["tb_ip"]
        # self.tb_token = self.config["tb_token"]
        
        # self.tb_gateway = TBGatewayMqttClient(
        #     host=self.config["tb_ip"],
        #     username=self.config["tb_token"],
        # )
        # self.device_name = f"RiscDetector - {self.config['device_id']}"
        
        # try:
        #     self.tb_gateway.connect()
        #     self.tb_gateway.gw_connect_device(self.device_name)
        #     self.tb_gateway.gw_send_attributes(f"RiscDetector - {self.config['device_id']}", {"Version": "detector_tb"})
        #     print("Conectado ao ThingsBoard.")
        # except Exception as e:
        #     print(f"Erro na conexão MQTT: {e}")

        # --- Thingsboard Status ---
        self.no_danger_frames = 0 
        self.status = False 

        # --- Video Capture ---
        self.cap = cv2.VideoCapture(self.config["video_path"][0])

        # --- ROI Handlers ---
        self.roi_handlers = []
        for coords in roi_data.values():
            roi = ROIHandler()
            roi.roi = tuple(coords.values())
            self.roi_handlers.append(roi)

    def on_control_message(self, client, userdata, msg):
        """
        Recebe ordens do Streamlit
        """
        try:
            print(f"📩 MENSAGEM RECEBIDA: {msg.payload}")
            payload = json.loads(msg.payload.decode())
            cmd = payload.get("command")
            
            if cmd == "START":
                cfg = payload.get("config", {})
                self.save_interval = cfg.get("save_interval", 5)
                self.max_saves = cfg.get("max_saves", 10)
                
                self.saved_count = 0
                self.last_save_time = time.time()
                self.parquet_buffer = [] 
                
                self.is_running = True 
                print(f"🟢 INICIANDO: {self.max_saves} batches a cada {self.save_interval}s")

            elif cmd == "STOP":
                self.is_running = False
                print("🔴 PARAGEM FORÇADA RECEBIDA")
                
        except Exception as e:
            print(f"Erro ao processar comando: {e}")

    def build_payload(self, detections):
        """
        Create structured detection JSON payload from list of detections
        """
        return {
            "raspberry_id": self.config["device_id"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "detections": detections
        }

    def save_parquet(self):
        if len(self.parquet_buffer) == 0:
            return

        detections_df = pd.DataFrame(self.parquet_buffer)
        if 'bbox_coords' in detections_df.columns:
            detections_df[['box_x1', 'box_y1', 'box_x2', 'box_y2']] = pd.DataFrame(
                detections_df['bbox_coords'].tolist(), 
                index=detections_df.index
            )
            detections_df = detections_df.drop(columns=['bbox_coords'])

        self.parquet_buffer = []
        return detections_df

    def send_tb(self, payload):
        try:
            self.tb_gateway.gw_send_telemetry(f"RiscDetector - {self.config['device_id']}", {"ts": int(round(time.time() * 1000)), "values": payload})
        except:
            pass

    def run(self):
        try:             
            payload = {"status": False, "random": random.random()}                    
            self.send_tb(payload)
            print("ENVIOU OFF INICIAL")
        except Exception as ee:
            print(f"EXCEPTION: {ee}")

        while True:
            if not self.is_running:
                self.cap.grab() 
                time.sleep(0.5) 
                continue
                
            else:
                ret, frame = self.cap.read()

                if not ret:
                    print("Fim do vídeo ou erro na câmera.")
                    break

                frame_resized = cv2.resize(frame, self.resolution)
                danger_detected = False
                detections = []
    
                detection_time_start = time.time()

                for roi_id, roi_handler in enumerate(self.roi_handlers, start=1): 
                    cropped = roi_handler.crop_frame(frame_resized)
                    if cropped is None or cropped.size == 0:
                        continue

                    results = self.model(cropped, verbose=False) 

                    for result in results:
                        boxes = result.boxes # bbox data
                        keypoints = result.keypoints # Pose data

                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            if cls_id != 0: 
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # centroid
                            conf = float(box.conf[0])

                            kpts = []
                            if keypoints is not None:
                                raw_kpts = keypoints.data[i].tolist() 
                                kpts = raw_kpts
                                danger_detected = True
                                det_data = {
                                    "ts": detection_time_start,
                                    "roi_id": roi_id,
                                    "label_type": "person",
                                    "confidence": conf,
                                    "bbox_coords": (x1, y1, x2, y2),
                                    "keypoints": kpts 
                                    }
                                detections.append(det_data)
                                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                ### ---- to PARQUET ---- ###
                                self.parquet_buffer.append(det_data)

                    # Adicionar default payload se não houve detecções
                    if not danger_detected:
                        default_entry = self.default_payload.copy()
                        default_entry["ts"] = detection_time_start
                        self.parquet_buffer.append(default_entry)

                    if time.time() - self.last_save_time >= self.save_interval and self.saved_count < self.max_saves:
                        detections_df = self.save_parquet()
                        self.labeller.process_batch(detections_df)
                        self.last_save_time = time.time()
                        self.saved_count += 1
                        if self.saved_count >= self.max_saves:
                            self.is_running = False
                    
                    print("DETECTIONS", detections)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    payload = self.build_payload(detections)
                    if danger_detected:
                        self.no_danger_frames = 0
                        self.status = True
                        try:
                            ii = 0
                            for detection in detections:
                                det_copy = detection.copy() # pra n alterar o buffer parquet
                                det_copy["status"] = True
                                det_copy["random"] = random.random()
                                det_copy["polygonID"] = ii
                                self.send_tb(det_copy)
                                ii += 1
                                print(f"⚠️ ENVIOU ON (Pessoa detetada ROI {detection['roi_id']})")
                        except Exception as ee:
                            print(f"EXCEPTION: {ee}")
                    else:
                        NN = 10
                        if (self.status==True and self.no_danger_frames < NN):
                            self.no_danger_frames += 1
                        else:
                            self.status = False
                            if (self.no_danger_frames==NN):
                                self.no_danger_frames += 1
                                try:             
                                    payload = {"status": False, "random": random.random()}                    
                                    self.send_tb(payload)
                                    print("ENVIOU OFF")
                                except Exception as ee:
                                    print(f"EXCEPTION: {ee}")
                    time.sleep(0.1)

    def release(self):
        self.save_parquet()
        self.cap.release()
        try:
            self.tb_gateway.gw_disconnect_device(self.device_name)
            self.tb_gateway.disconnect()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config_paths = {
        "config": "client/config.yaml"
    }
    roi_path = ROIS_CONFIG_PATH
    model_params = {
        "model_path": "models/yolo11n-pose.pt", 
    }

    detector = StereoDetector(config_paths, roi_path, model_params)
    try:
        detector.run()
    finally:
        detector.release()