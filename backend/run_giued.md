to run the backend server + vm 


cd /home/zedny/Desktop/pathira/fullstack_proect
source backend/venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

