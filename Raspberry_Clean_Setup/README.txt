Pathira Raspberry clean OS setup
================================

Use Raspberry Pi Imager settings:
- Hostname: pathira.local
- Username: pathira
- Password: ziad123456789-Zz
- Enable SSH: yes
- WiFi SSID: pathira
- WiFi password: ziad123456789-Zz
- WiFi country: EG

After first boot:
1. Connect Android phone to the same WiFi: pathira / ziad123456789-Zz
2. Open Pathira APK -> Vision -> Play -> Glasses Camera
3. Copy this folder to Raspberry Pi as: ~/pathira_raspberry_setup
4. Run:

   cd ~/pathira_raspberry_setup
   chmod +x install_pathira_raspberry.sh
   bash install_pathira_raspberry.sh
   sudo systemctl start pathira-raspberry-camera.service

Logs:

   sudo journalctl -u pathira-raspberry-camera.service -f

Manual test:

   python3 ~/pathira_raspberry/pathira_raspberry_client.py --phone-host auto --phone-port 8080

If auto fails, the app Debug card shows http://PHONE_IP:8080, then run:

   python3 ~/pathira_raspberry/pathira_raspberry_client.py --phone-host PHONE_IP --phone-port 8080
