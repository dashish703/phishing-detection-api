import datetime
import json
import os

class Firewall:
    def __init__(self, log_file="phishing_log.json"):
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, "w") as file:
                json.dump([], file)  # Create empty JSON log file

    def log_phishing_attempt(self, email_text, url, prediction):
        """
        Logs phishing detection attempts with timestamp.
        """
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email_text": email_text[:100] + "..." if len(email_text) > 100 else email_text,
            "url": url,
            "prediction": "Phishing" if prediction == 1 else "Safe"
        }

        # Append to JSON log file
        with open(self.log_file, "r+") as file:
            logs = json.load(file)
            logs.append(log_entry)
            file.seek(0)
            json.dump(logs, file, indent=4)

        print(f"🔍 [Firewall] Logged phishing attempt: {log_entry}")

    def check_recent_attempts(self):
        """
        Reads and prints recent phishing attempts.
        """
        if not os.path.exists(self.log_file):
            print("⚠️ No logs found!")
            return

        with open(self.log_file, "r") as file:
            logs = json.load(file)
            print("\n🔍 Recent Phishing Detection Logs:")
            for entry in logs[-5:]:  # Show last 5 attempts
                print(entry)

# Example usage:
if __name__ == "__main__":
    firewall = Firewall()
    firewall.log_phishing_attempt("Fake bank email", "http://phishy.com", 1)
    firewall.check_recent_attempts()
