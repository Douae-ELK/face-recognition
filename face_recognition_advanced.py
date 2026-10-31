import cv2
import numpy as np
import pickle
import os
import sqlite3
import csv
from datetime import datetime, date
import face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "face_data")
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
DB_PATH = os.path.join(DATA_DIR, "face_database.db")
ENCODINGS_PATH = os.path.join(DATA_DIR, "face_encodings.pkl")

CATEGORY_COLORS = {
    "Student":              (0, 255, 0),    
    "Professor":            (0, 215, 255),  
    "Administrative Staff": (255, 165, 0),  
    "Unknown":              (0, 0, 255),    
}


class FaceRecognition:
    def __init__(self):
        self.db_path = DB_PATH
        self.encodings_path = ENCODINGS_PATH

        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_categories = []

        self.frame_width = 1280
        self.frame_height = 720

        self.face_locations = []

        self.training_mode = False
        self.training_name = ""
        self.training_category = ""
        self.training_samples = []
        self.samples_needed = 10
        self.category_selection_mode = False

        self.recognition_threshold = 0.6
        self.process_every_n_frames = 2

        self._last_raw = None
        self._last_face_loc = None
        self._last_unknown_alert = None
        self._unknown_cooldown_seconds = 10
        self._cached_results = []
        self._cached_train_locs = []

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(PHOTOS_DIR, exist_ok=True)

        self._init_database()
        self._load_encodings()

        print("✓ Face Recognition System initialized!")
        print(f"  Known faces: {len(self.known_face_names)}")
        print(f"  Data dir: {DATA_DIR}/")

    # ------------------------------------------------------------------ DB --

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT DEFAULT "Student",
                photo_path TEXT,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP,
                times_recognized INTEGER DEFAULT 0
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                date DATE,
                first_seen TEXT,
                last_seen TEXT,
                total_appearances INTEGER DEFAULT 1,
                UNIQUE(person_id, date),
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS unknown_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                photo_path TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def _add_person(self, name, category, photo_path=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                'INSERT INTO persons (name, category, photo_path) VALUES (?, ?, ?)',
                (name, category, photo_path)
            )
            person_id = c.lastrowid
            conn.commit()
            return person_id
        except sqlite3.IntegrityError:
            c.execute('SELECT id FROM persons WHERE name = ?', (name,))
            return c.fetchone()[0]
        finally:
            conn.close()

    def _update_recognition(self, person_id, confidence):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        today = date.today().isoformat()
        now_time = datetime.now().strftime('%H:%M:%S')

        c.execute('''
            UPDATE persons
            SET last_seen = CURRENT_TIMESTAMP, times_recognized = times_recognized + 1
            WHERE id = ?
        ''', (person_id,))

        c.execute('INSERT INTO recognitions (person_id, confidence) VALUES (?, ?)',
                  (person_id, confidence))

        c.execute('''
            INSERT INTO attendance (person_id, date, first_seen, last_seen, total_appearances)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(person_id, date) DO UPDATE SET
                last_seen = excluded.last_seen,
                total_appearances = total_appearances + 1
        ''', (person_id, today, now_time, now_time))

        conn.commit()
        conn.close()

    # --------------------------------------------------------- Encodings --

    def _save_encodings(self):
        data = {
            'encodings':  self.known_face_encodings,
            'names':      self.known_face_names,
            'ids':        self.known_face_ids,
            'categories': self.known_face_categories,
        }
        with open(self.encodings_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_encodings(self):
        if not os.path.exists(self.encodings_path):
            print("  No saved encodings found — train some faces first!")
            return
        with open(self.encodings_path, 'rb') as f:
            data = pickle.load(f)
        self.known_face_encodings = data['encodings']
        self.known_face_names = data['names']
        self.known_face_ids = data['ids']
        self.known_face_categories = data.get('categories',
                                              ['Student'] * len(data['names']))
        print(f"  Loaded {len(self.known_face_names)} encodings from disk")

    # --------------------------------------------------------- Recognition --

    def _recognize_faces(self, raw_frame):
        small = cv2.resize(raw_frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        self.face_locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, self.face_locations)

        results = []
        for enc, loc in zip(encodings, self.face_locations):
            name, confidence, category, person_id = "Unknown", 0.0, "Unknown", None

            if self.known_face_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, enc)
                best = int(np.argmin(distances))
                conf = max(0.0, float(1 - distances[best]))
                if distances[best] < self.recognition_threshold:
                    name = self.known_face_names[best]
                    confidence = conf
                    category = self.known_face_categories[best]
                    person_id = self.known_face_ids[best]
                    self._update_recognition(person_id, confidence)

            top, right, bottom, left = loc
            scaled = (top * 4, right * 4, bottom * 4, left * 4)

            if name == "Unknown":
                self._maybe_save_unknown(raw_frame, scaled)

            results.append((name, confidence, category, scaled))

        return results

    def _maybe_save_unknown(self, frame, location):
        now = datetime.now()
        if self._last_unknown_alert:
            if (now - self._last_unknown_alert).seconds < self._unknown_cooldown_seconds:
                return
        self._last_unknown_alert = now

        top, right, bottom, left = location
        pad = 20
        face_img = frame[max(0, top - pad):bottom + pad,
                         max(0, left - pad):right + pad]
        if face_img.size == 0:
            return

        ts = now.strftime('%Y%m%d_%H%M%S')
        photo_path = os.path.join(PHOTOS_DIR, f"unknown_{ts}.jpg")
        cv2.imwrite(photo_path, face_img)

        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT INTO unknown_alerts (photo_path) VALUES (?)', (photo_path,))
        conn.commit()
        conn.close()

        print(f"  Unknown face saved: {photo_path}")

    # ----------------------------------------------------------- Training --

    def _capture_sample(self, rgb_frame, face_location):
        enc = face_recognition.face_encodings(rgb_frame, [face_location])
        if enc:
            self.training_samples.append(enc[0])
            return True
        return False

    def _finalize_training(self):
        if len(self.training_samples) < 3:
            print("  Not enough samples (need at least 3).")
            return False

        avg_enc = np.mean(self.training_samples, axis=0)

        photo_path = None
        if self._last_raw is not None and self._last_face_loc is not None:
            top, right, bottom, left = self._last_face_loc
            pad = 20
            face_img = self._last_raw[max(0, top - pad):bottom + pad,
                                      max(0, left - pad):right + pad]
            if face_img.size > 0:
                safe = ''.join(
                    ch if ch.isalnum() or ch in ('-', '_') else '_'
                    for ch in self.training_name
                ).strip('_') or 'person'
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                photo_path = os.path.join(PHOTOS_DIR, f"{safe}_{ts}.jpg")
                cv2.imwrite(photo_path, face_img)

        person_id = self._add_person(self.training_name, self.training_category, photo_path)

        self.known_face_encodings.append(avg_enc)
        self.known_face_names.append(self.training_name)
        self.known_face_ids.append(person_id)
        self.known_face_categories.append(self.training_category)

        self._save_encodings()
        print(f"✓ Trained: {self.training_name} ({self.training_category})"
              f" — {len(self.training_samples)} samples")
        return True

    def _reset_training(self):
        self.training_mode = False
        self.training_name = ""
        self.training_category = ""
        self.training_samples = []
        self.category_selection_mode = False
        self._last_raw = None
        self._last_face_loc = None

    # ----------------------------------------------------------------- UI --

    def _draw_box(self, frame, name, confidence, category, location):
        top, right, bottom, left = location
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"])

        if name == "Unknown":
            label = "Unknown"
        else:
            label = f"{name} [{category}] {confidence:.0%}"

        # Widen the label background if the text is longer than the face box
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_right = max(right, left + text_w + 12)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (label_right, bottom), color, cv2.FILLED)

        cv2.putText(frame, label, (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_ui(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.training_mode and self.category_selection_mode:
            cv2.putText(frame,
                        f"SELECT CATEGORY for: {self.training_name}",
                        (20, 40), font, 0.85, (0, 165, 255), 2)
            cv2.putText(frame,
                        "S = Student   P = Professor   A = Administrative Staff   ESC = Cancel",
                        (20, 80), font, 0.55, (200, 200, 200), 1)

        elif self.training_mode:
            cv2.putText(frame,
                        f"TRAINING: {self.training_name}  [{self.training_category}]",
                        (20, 40), font, 0.85, (0, 165, 255), 2)
            cv2.putText(frame,
                        f"Samples: {len(self.training_samples)}/{self.samples_needed}",
                        (20, 75), font, 0.65, (0, 255, 0), 2)
            cv2.putText(frame,
                        "SPACE = Capture   ENTER = Finish   ESC = Cancel",
                        (20, 108), font, 0.5, (200, 200, 200), 1)

        else:
            cv2.putText(frame, "FACE RECOGNITION",
                        (20, 42), font, 0.9, (0, 255, 255), 2)
            status = (f"Known: {len(self.known_face_names)}  |  "
                      f"Detected: {len(self.face_locations)}")
            cv2.putText(frame, status, (20, 78), font, 0.6, (255, 255, 255), 1)
            cv2.putText(frame,
                        "N = New face   T = Stats   E = Export CSV   Q = Quit",
                        (20, 108), font, 0.5, (200, 200, 200), 1)

        return frame

    # --------------------------------------------------------- Stats / CSV --

    def show_statistics(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT name, category, date_added, last_seen, times_recognized
            FROM persons ORDER BY times_recognized DESC
        ''')
        rows = c.fetchall()

        print("\n" + "=" * 80)
        print("PERSONS")
        print("=" * 80)
        if rows:
            print(f"{'Name':<22} {'Cat':<10} {'Added':<20} {'Last Seen':<20} {'Times'}")
            print("-" * 80)
            for name, cat, added, last_seen, times in rows:
                print(f"{name:<22} {cat:<10} {(added or '')[:19]:<20} "
                      f"{(last_seen or 'Never')[:19]:<20} {times}")
        else:
            print("No persons yet.")

        today = date.today().isoformat()
        c.execute('''
            SELECT p.name, p.category, a.first_seen, a.last_seen, a.total_appearances
            FROM attendance a JOIN persons p ON a.person_id = p.id
            WHERE a.date = ? ORDER BY a.first_seen
        ''', (today,))
        att = c.fetchall()

        print(f"\nATTENDANCE  ({today})")
        print("-" * 80)
        if att:
            for name, cat, first, last, total in att:
                print(f"  {name} ({cat})  |  In: {first}  Out: {last}  |  {total}x")
        else:
            print("  No attendance records today.")

        print("=" * 80 + "\n")
        conn.close()

    def export_report(self):
        today = date.today().isoformat()
        filename = os.path.join(BASE_DIR, f"attendance_{today}.csv")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT p.name, p.category, a.first_seen, a.last_seen, a.total_appearances
            FROM attendance a JOIN persons p ON a.person_id = p.id
            WHERE a.date = ? ORDER BY a.first_seen
        ''', (today,))
        rows = c.fetchall()
        conn.close()

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Category', 'First Seen', 'Last Seen', 'Total Appearances'])
            writer.writerows(rows)

        print(f"✓ Report exported → {filename}  ({len(rows)} records)")

    # ------------------------------------------------------------ Main loop --

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened. Check webcam permissions and availability.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        print("\n" + "=" * 60)
        print("FACE RECOGNITION — STARTED")
        print("=" * 60)
        print("Boxes:  Green=Student | Gold=Professor | Blue=Administrative Staff | Red=Unknown")
        print("Keys:   N=New face | T=Stats | E=Export CSV | Q=Quit")
        print("=" * 60 + "\n")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            frame_count += 1

            # --- Process every Nth frame ---
            if frame_count % self.process_every_n_frames == 0:
                if not self.training_mode:
                    self._cached_results = self._recognize_faces(raw_frame)
                else:
                    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    self._cached_train_locs = face_recognition.face_locations(rgb)
                    self.face_locations = self._cached_train_locs

            # --- Draw ---
            if not self.training_mode:
                for name, conf, cat, loc in self._cached_results:
                    self._draw_box(frame, name, conf, cat, loc)
            else:
                for top, right, bottom, left in self._cached_train_locs:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                    if not self.category_selection_mode:
                        cv2.putText(frame, "Training...", (left, top - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            frame = self._draw_ui(frame)
            cv2.imshow('Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF

            # --- Quit ---
            if key == ord('q'):
                break

            # --- Start training ---
            elif key == ord('n') and not self.training_mode:
                if not self.face_locations:
                    print("  No face detected — position face in view first.")
                else:
                    name = input("\nEnter person's name: ").strip()
                    if name:
                        self.training_name = name
                        self.training_samples = []
                        self.category_selection_mode = True
                        self.training_mode = True
                        print("Select category:  S = Student   P = Professor   A = Administrative Staff")
                    else:
                        print("  Invalid name — training cancelled.")

            # --- Category selection ---
            elif self.training_mode and self.category_selection_mode:
                cat_map = {ord('s'): 'Student', ord('p'): 'Professor', ord('a'): 'Administrative Staff'}
                if key in cat_map:
                    self.training_category = cat_map[key]
                    self.category_selection_mode = False
                    print(f"  Category: {self.training_category}"
                          f"  — press SPACE to capture {self.samples_needed} samples")
                elif key == 27:
                    self._reset_training()
                    print("  Training cancelled.")

            # --- Capture sample ---
            elif key == ord(' ') and self.training_mode and not self.category_selection_mode:
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb)
                if face_locs:
                    ok = self._capture_sample(rgb, face_locs[0])
                    if ok:
                        self._last_raw = raw_frame.copy()
                        self._last_face_loc = face_locs[0]
                        print(f"  Sample {len(self.training_samples)}/{self.samples_needed} captured")
                        if len(self.training_samples) >= self.samples_needed:
                            print("  Enough samples — press ENTER to finish.")
                else:
                    print("  No face detected!")

            # --- Finish training ---
            elif key == 13 and self.training_mode and not self.category_selection_mode:
                if self._finalize_training():
                    self._reset_training()
                else:
                    print("  Keep capturing more samples.")

            # --- Cancel training ---
            elif key == 27 and self.training_mode:
                self._reset_training()
                print("  Training cancelled.")

            # --- Stats / export ---
            elif key == ord('t'):
                self.show_statistics()

            elif key == ord('e'):
                self.export_report()

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Face Recognition System closed")


if __name__ == "__main__":
    try:
        system = FaceRecognition()
        system.run()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()