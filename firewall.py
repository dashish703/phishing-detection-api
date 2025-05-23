import datetime
import json
import os
from typing import List, Dict


class Firewall:
    def __init__(self, log_file: str = "phishing_log.json"):
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, "w") as file:
                json.dump([], file)  # Create empty JSON log file

    def log_phishing_attempt(self, email_text: str, url: str, prediction: int) -> None:
        """
        Logs phishing detection attempts with timestamp.
        """
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email_text": (email_text[:100] + "...") if len(email_text) > 100 else email_text,
            "url": url,
            "prediction": "Phishing" if prediction == 1 else "Safe"
        }

        try:
            with open(self.log_file, "r+", encoding="utf-8") as file:
                logs = json.load(file)
                logs.append(log_entry)
                file.seek(0)
                json.dump(logs, file, indent=4)
            print(f"🔍 [Firewall] Logged phishing attempt:\n{json.dumps(log_entry, indent=4)}")
        except Exception as e:
            print(f"❌ [Firewall Error] Failed to log entry: {e}")

    def check_recent_attempts(self) -> None:
        """
        Reads and prints recent phishing attempts.
        """
        if not os.path.exists(self.log_file):
            print("⚠️ No logs found!")
            return

        try:
            with open(self.log_file, "r", encoding="utf-8") as file:
                logs = json.load(file)
                print("\n🔍 Recent Phishing Detection Logs:")
                for entry in logs[-5:]:
                    print(json.dumps(entry, indent=4))
        except Exception as e:
            print(f"❌ [Firewall Error] Could not read logs: {e}")

    def get_recent_attempts(self, count: int = 5) -> List[Dict]:
        """
        Returns recent phishing detection attempts.
        """
        if not os.path.exists(self.log_file):
            return []

        try:
            with open(self.log_file, "r", encoding="utf-8") as file:
                logs = json.load(file)
                return logs[-count:]
        except Exception as e:
            print(f"❌ [Firewall Error] Failed to retrieve logs: {e}")
            return []


# Example usage
if __name__ == "__main__":
    firewall = Firewall()
    firewall.log_phishing_attempt("Fake bank email", "http://phishy.com", 1)
    firewall.check_recent_attempts()
