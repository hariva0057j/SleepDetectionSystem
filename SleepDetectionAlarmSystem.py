import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame
import threading
import json
import os
from collections import deque
from scipy.spatial import distance as dist


class MLEnhancedSleepDetectionSystem:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize pygame mixer for alarm sounds
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Camera parameters
        self.cam_width = 640
        self.cam_height = 480
        
        # Eye landmarks to compute EAR
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        
        # Sleep detection parameters
        self.EAR_THRESHOLD = 0.20
        self.default_alarm_threshold = 5.0  # seconds
        self.adaptive_threshold = self.default_alarm_threshold
        self.min_threshold = 2.0
        self.max_threshold = 10.0
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_score = 0.5
        self.user_profile = {
            'normal_blink_duration': [],
            'normal_blink_frequency': [],
            'false_alarms': 0,
            'successful_alarms': 0,
            'total_sessions': 0
        }
        
        # Optimized data buffers with fixed sizes
        self.blink_history = deque(maxlen=50)
        self.ear_history = deque(maxlen=100)
        
        # Tracking stats
        self.eyes_closed_start = None
        self.sleep_level = 0  # 0: Alert, 1: Sleep detected
        self.sleep_probability = 0.0
        self.last_alarm_reason = ""
        
        # Thread-safe alarm state
        self.alarm_active = False
        self.alarm_thread = None
        self._alarm_lock = threading.Lock()
        
        # Session stats
        self.total_blinks = 0
        self.session_start_time = time.time()
        
        # Load user profile if exists
        self.load_user_profile()

    def calculate_ear(self, landmarks, eye_points):
        """Calculate Eye Aspect Ratio (EAR) - Optimized version"""
        # Direct coordinate extraction without intermediate arrays
        coords = [(landmarks[i].x * self.cam_width, landmarks[i].y * self.cam_height) for i in eye_points]
        
        # Vectorized distance calculations
        vertical_1 = dist.euclidean(coords[1], coords[5])
        vertical_2 = dist.euclidean(coords[2], coords[4])
        horizontal = dist.euclidean(coords[0], coords[3])
        
        return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal > 0 else 0

    def detect_drowsiness_onset(self):
        """Detect early drowsiness patterns - Optimized"""
        if len(self.ear_history) < 20:
            return False

        # Get recent EAR values efficiently
        recent_ears = [e['ear'] for e in list(self.ear_history)[-20:]]
        
        # Calculate trend efficiently
        if len(recent_ears) <= 1:
            return False
            
        ear_trend = np.polyfit(range(len(recent_ears)), recent_ears, 1)[0]
        
        # Check blink variance only if we have data
        current_time = time.time()
        recent_blinks = [b for b in self.blink_history if b['timestamp'] > current_time - 30]
        
        if recent_blinks and self.user_profile['normal_blink_duration']:
            blink_variance = np.var([b['duration'] for b in recent_blinks])
            normal_variance = np.mean(self.user_profile['normal_blink_duration'])
            return blink_variance > normal_variance * 1.5 and ear_trend < -0.001
        
        return False

    def predict_sleep_probability(self, current_ear, eyes_closed_duration):
        """Predict sleep probability - Optimized calculation"""
        indicators = []
        
        # Factor 1: EAR deviation
        if self.user_profile['normal_blink_duration']:
            normal_ear = np.mean(self.user_profile['normal_blink_duration'])
            ear_deviation = abs(current_ear - normal_ear) / (normal_ear + 0.1)
            indicators.append(min(ear_deviation, 1.0))
        else:
            indicators.append(max(0, (self.EAR_THRESHOLD - current_ear) / self.EAR_THRESHOLD))
        
        # Factor 2: Duration factor
        indicators.append(min(eyes_closed_duration / self.adaptive_threshold, 1.0))
        
        # Factor 3: Time of day factor (optimized lookup)
        hour = time.localtime().tm_hour
        sleepy_hours = {1: 0.8, 2: 0.9, 3: 0.95, 4: 0.9, 5: 0.7, 13: 0.6, 14: 0.7, 15: 0.6}
        indicators.append(sleepy_hours.get(hour, 0.3))
        
        # Factor 4: Drowsiness pattern
        indicators.append(0.8 if self.detect_drowsiness_onset() else 0.2)
        
        # Weighted average calculation
        weights = [0.3, 0.4, 0.15, 0.15]
        return min(np.average(indicators, weights=weights), 1.0)

    def learn_user_patterns(self, session_outcome='neutral'):
        """Update user profile - Optimized learning"""
        # Process normal blinks efficiently
        normal_blinks = [b['duration'] for b in self.blink_history if b['duration'] < 1.0]
        if normal_blinks:
            avg_blink = np.mean(normal_blinks)
            self.user_profile['normal_blink_duration'].append(avg_blink)
            # Keep only recent data (memory optimization)
            if len(self.user_profile['normal_blink_duration']) > 20:
                self.user_profile['normal_blink_duration'] = self.user_profile['normal_blink_duration'][-20:]
        
        # Update stats and threshold
        if session_outcome == 'false_alarm':
            self.user_profile['false_alarms'] += 1
            self._adjust_threshold(True)
            print("🧠 False alarm - Adjusting threshold up")
        elif session_outcome == 'successful_wake':
            self.user_profile['successful_alarms'] += 1
            self._adjust_threshold(False)
            print("🧠 Successful wake-up - Adjusting threshold down")
        
        self.user_profile['total_sessions'] += 1

    def _adjust_threshold(self, increase):
        """Adjust threshold with boundary checking"""
        adjustment = 0.5 * self.learning_rate
        old_threshold = self.adaptive_threshold
        
        if increase:
            self.adaptive_threshold = min(self.adaptive_threshold + adjustment, self.max_threshold)
        else:
            self.adaptive_threshold = max(self.adaptive_threshold - adjustment, self.min_threshold)
        
        if self.adaptive_threshold != old_threshold:
            print(f"🧠 Threshold: {old_threshold:.1f}s → {self.adaptive_threshold:.1f}s")
        
        # Update confidence score
        total_alarms = self.user_profile['false_alarms'] + self.user_profile['successful_alarms']
        self.confidence_score = self.user_profile['successful_alarms'] / total_alarms if total_alarms > 0 else 0.5

    def intelligent_sleep_analysis(self, ear):
        """Main sleep analysis with optimized logic"""
        current_time = time.time()
        
        # Add to history efficiently
        self.ear_history.append({'ear': ear, 'timestamp': current_time, 'alert_level': self.sleep_level})
        
        # Calculate eyes closed duration
        if ear < self.EAR_THRESHOLD:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
                print(f"👁️ Eyes closed (EAR: {ear:.3f})")
            eyes_closed_duration = current_time - self.eyes_closed_start
        else:
            if self.eyes_closed_start is not None:
                closed_duration = current_time - self.eyes_closed_start
                if closed_duration < 1.0:  # Normal blink
                    self.total_blinks += 1
                    self.blink_history.append({
                        'duration': closed_duration, 
                        'timestamp': current_time, 
                        'ear_before': ear
                    })
                print(f"👁️ Eyes opened (closed {closed_duration:.2f}s)")
            self.eyes_closed_start = None
            eyes_closed_duration = 0
        
        # Get sleep probability
        self.sleep_probability = self.predict_sleep_probability(ear, eyes_closed_duration)
        
        # Determine alarm trigger - Optimized decision logic
        should_alarm, alarm_reason = self._should_trigger_alarm(eyes_closed_duration)
        
        if should_alarm:
            self.sleep_level = 1
            self.last_alarm_reason = alarm_reason
            if not self.alarm_active:
                print(f"🚨 ALARM: {alarm_reason}")
        else:
            self.sleep_level = 0
        
        return eyes_closed_duration

    def _should_trigger_alarm(self, eyes_closed_duration):
        """Optimized alarm decision logic"""
        if eyes_closed_duration > self.adaptive_threshold:
            return True, f"Eyes closed {eyes_closed_duration:.1f}s > threshold {self.adaptive_threshold:.1f}s"
        
        if self.detect_drowsiness_onset() and self.sleep_probability > 0.7:
            return True, f"Early drowsiness (prob {self.sleep_probability:.2f})"
        
        if self.sleep_probability > 0.85 and eyes_closed_duration > 2.0:
            return True, f"High sleep probability {self.sleep_probability:.2f}"
        
        return False, ""

    def get_user_feedback(self):
        """Simplified user feedback"""
        print("\n" + "="*50)
        print("🧠 FEEDBACK: (h)elpful | (f)alse alarm | (l)ate | (s)kip")
        
        try:
            feedback = input("Choice: ").lower().strip()
            feedback_actions = {
                'h': lambda: self.learn_user_patterns('successful_wake'),
                'f': lambda: self.learn_user_patterns('false_alarm'),
                'l': lambda: setattr(self, 'adaptive_threshold', max(self.adaptive_threshold - 0.5, self.min_threshold))
            }
            
            if feedback in feedback_actions:
                feedback_actions[feedback]()
                print("✅ Feedback processed!")
            else:
                print("⏭️ Feedback skipped.")
        except Exception:
            print("⏭️ Input error - feedback skipped.")

    def save_user_profile(self, filename="user_sleep_profile.json"):
        """Optimized profile saving"""
        profile_data = {
            'adaptive_threshold': self.adaptive_threshold,
            'confidence_score': self.confidence_score,
            'user_stats': self.user_profile,
            'last_updated': time.time(),
            'version': '2.1'
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(profile_data, f, indent=2)
            print(f"💾 Profile saved! Threshold: {self.adaptive_threshold:.1f}s")
        except Exception as e:
            print(f"❌ Save failed: {e}")

    def load_user_profile(self, filename="user_sleep_profile.json"):
        """Optimized profile loading"""
        try:
            if not os.path.exists(filename):
                print("🆕 No profile found. Starting fresh.")
                return
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.adaptive_threshold = data.get('adaptive_threshold', self.default_alarm_threshold)
            self.confidence_score = data.get('confidence_score', 0.5)
            
            # Merge loaded stats
            loaded_stats = data.get('user_stats', {})
            for key, value in loaded_stats.items():
                if key in self.user_profile:
                    self.user_profile[key] = value
            
            sessions = self.user_profile.get('total_sessions', 0)
            print(f"📂 Profile loaded! Threshold: {self.adaptive_threshold:.1f}s ({sessions} sessions)")
            
        except Exception as e:
            print(f"⚠️ Load error: {e}. Using defaults.")

    def generate_alarm_sound(self, frequency=1000, duration=0.8, volume=0.8):
        """Optimized alarm sound generation"""
        sample_rate = 22050
        frames = int(duration * sample_rate)
        t = np.linspace(0, duration, frames)
        
        # Generate combined waveform efficiently
        wave = volume * (np.sin(2 * np.pi * frequency * t) + 
                        0.5 * np.sin(2 * np.pi * frequency * 1.5 * t))
        
        # Convert to stereo
        stereo_wave = np.column_stack((wave, wave))
        audio_data = (stereo_wave * 16383).astype(np.int16)
        
        return pygame.sndarray.make_sound(audio_data)

    def start_alarm(self):
        """Thread-safe alarm start"""
        with self._alarm_lock:
            if not self.alarm_active and self.sleep_level == 1:
                self.alarm_active = True
                print("🚨 ALARM STARTED - WAKE UP! 🚨")
                self.alarm_thread = threading.Thread(target=self._alarm_loop, daemon=True)
                self.alarm_thread.start()

    def stop_alarm(self):
        """Thread-safe alarm stop"""
        with self._alarm_lock:
            if self.alarm_active:
                self.alarm_active = False
                if self.alarm_thread and self.alarm_thread.is_alive():
                    self.alarm_thread.join(timeout=1)
                print("🔇 Alarm stopped")

    def _alarm_loop(self):
        """Optimized alarm loop"""
        while self.alarm_active and self.sleep_level == 1:
            for i in range(3):
                if not self.alarm_active:
                    break
                frequency = 1000 + (i * 200)
                sound = self.generate_alarm_sound(frequency, 0.8, 0.8)
                sound.play()
                time.sleep(0.3)
            time.sleep(0.7)

    def draw_status(self, frame, ear, eyes_closed_duration):
        """Optimized status drawing"""
        # Status text
        if self.sleep_level == 1:
            status_text, color = "⚠️ ALARM ACTIVE - WAKE UP! ⚠️", (0, 0, 255)
        else:
            status_text, color = "🧠 ML-MONITORING - ALERT", (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # EAR display
        ear_color = (0, 0, 255) if ear < self.EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        
        # Eyes closed duration with progress bar
        if eyes_closed_duration > 0:
            duration_color = (0, 0, 255) if eyes_closed_duration > self.adaptive_threshold * 0.8 else (255, 255, 0)
            cv2.putText(frame, f"Closed: {eyes_closed_duration:.1f}s / {self.adaptive_threshold:.1f}s", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, duration_color, 2)
            
            # Progress bar
            progress = min(eyes_closed_duration / self.adaptive_threshold, 1.0)
            bar_color = (0, 255, 255) if progress < 0.8 else (0, 0, 255)
            cv2.rectangle(frame, (10, 130), (310, 150), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 130), (10 + int(300 * progress), 150), bar_color, -1)
            cv2.putText(frame, f"{progress*100:.0f}%", (320, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ML info
        y = 170
        prob_color = (0, 255, 0) if self.sleep_probability < 0.5 else (0, 255, 255) if self.sleep_probability < 0.8 else (0, 0, 255)
        cv2.putText(frame, f"💤 Sleep Prob: {self.sleep_probability:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, prob_color, 2)
        cv2.putText(frame, f"🧠 Threshold: {self.adaptive_threshold:.1f}s (Conf: {self.confidence_score:.2f})", (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Session info
        session_duration = (time.time() - self.session_start_time) / 60
        cv2.putText(frame, f"Session: {session_duration:.1f}min | Blinks: {self.total_blinks}", (10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(frame, "q=quit | r=reset | s=stop | f=feedback | p=save", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self):
        """Optimized main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🤖 ML-Enhanced Sleep Detection System v2.1 🤖")
        print("=" * 50)
        print(f"🧠 Adaptive threshold: {self.adaptive_threshold:.1f}s")
        print("Controls: f=feedback | p=save | r=reset | s=stop | q=quit")
        print("=" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Calculate EAR
                    left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_LANDMARKS)
                    right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_LANDMARKS)
                    ear = (left_ear + right_ear) / 2
                    
                    # Analyze sleep
                    eyes_closed_duration = self.intelligent_sleep_analysis(ear)
                    
                    # Handle alarm
                    if self.sleep_level == 1:
                        self.start_alarm()
                    else:
                        self.stop_alarm()
                    
                    # Draw status
                    self.draw_status(frame, ear, eyes_closed_duration)
                    
                    # Draw eye landmarks
                    for idx in self.LEFT_EYE_LANDMARKS + self.RIGHT_EYE_LANDMARKS:
                        x, y = int(landmarks[idx].x * self.cam_width), int(landmarks[idx].y * self.cam_height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                else:
                    cv2.putText(frame, "❌ No Face Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.stop_alarm()
                
                cv2.imshow('ML Sleep Detection v2.1', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_session()
                elif key == ord('s'):
                    self.stop_alarm()
                elif key == ord('f') and self.last_alarm_reason:
                    self.get_user_feedback()
                elif key == ord('p'):
                    self.save_user_profile()
        
        except KeyboardInterrupt:
            print("\n⚠️ System interrupted")
        finally:
            self.stop_alarm()
            self.save_user_profile()
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print("🔚 System shutdown complete!")

    def _reset_session(self):
        """Reset session statistics"""
        self.total_blinks = 0
        self.session_start_time = time.time()
        self.blink_history.clear()
        self.ear_history.clear()
        print("📊 Session reset!")


if __name__ == "__main__":
    try:
        system = MLEnhancedSleepDetectionSystem()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
