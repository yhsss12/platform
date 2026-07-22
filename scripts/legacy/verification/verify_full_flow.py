"""Legacy API smoke script; retained for reference, not current acceptance."""

import urllib.request
import urllib.parse
import json
import uuid
import time
import sys

BASE_URL = "http://127.0.0.1:8000/api"
MCAP_PATH = "/tmp/test_job_data.mcap"

def request(method, url, data=None, headers=None):
    if headers is None:
        headers = {}
    
    if data is not None:
        data_bytes = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    else:
        data_bytes = None
    
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))
    except Exception as e:
        print(f"Request failed: {e}")
        return 500, {}

def main():
    # Ensure a dummy mcap file exists for validation
    with open(MCAP_PATH, "wb") as f:
        f.write(b"dummy mcap content")

    # 1. Login
    print("Logging in...")
    status, body = request("POST", f"{BASE_URL}/auth/login", {"username": "admin", "password": "password"})
    
    if status != 200 or not body.get("ok"):
         print("Retrying with admin123...")
         status, body = request("POST", f"{BASE_URL}/auth/login", {"username": "admin", "password": "admin123"})
    
    if status != 200 or not body.get("ok"):
         print(f"Login failed: {status} {body}")
         return

    token = body["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("Login successful")

    # 2. Create a Job
    task_id = str(uuid.uuid4())
    create_payload = {
        "task_id": task_id,
        "operator_name": "tester",
        "status": "PENDING",
        "collection_quantity": 10,
        "completed_count": 0
    }
    print("Creating job...")
    status, body = request("POST", f"{BASE_URL}/jobs", create_payload, headers)
    if status != 200:
        print(f"Create Job failed: {status} {body}")
        return
    
    job_data = body["data"]
    job_id = job_data["id"]
    print(f"Job created: {job_id}")

    # 3. Update Job (Simulate SaveDataDialog)
    # progress: {current: 1, total: 10}, mcap_path
    update_payload = {
        "progress": {"current": 1, "total": 10},
        "mcap_path": MCAP_PATH,
        "register_collect_asset": True,
        "mcap_size_bytes": 1024,
        "status": "RUNNING"
    }
    
    print("Updating job...")
    status, body = request("PATCH", f"{BASE_URL}/jobs/{job_id}", update_payload, headers)
    if status != 200:
        print(f"Update Job failed: {status} {body}")
        return
    
    updated_job = body["data"]
    print(f"Job updated response: progress={updated_job.get('progress')}, completed={updated_job.get('completed_count')}")
    
    if updated_job.get('completed_count') != 1:
        print("FAILURE: completed_count not updated in response")
    else:
        print("SUCCESS: Job progress updated in response")

    # 4. Verify Data Asset
    print("Verifying HDF5 dataset...")
    time.sleep(1)
    
    status, body = request("GET", f"{BASE_URL}/hdf5-datasets", None, headers)
    if status != 200:
        print(f"List Datasets failed: {status} {body}")
        return
    
    datasets = body["data"]["items"]
    found = False
    for ds in datasets:
        # Check if storage_uri matches MCAP_PATH or file_path matches
        if ds.get("storage_uri") == MCAP_PATH or ds.get("file_path") == MCAP_PATH:
            found = True
            print(f"Found Dataset: {ds['id']} - {ds['name']}")
            break
    
    if found:
        print("SUCCESS: Data Asset registered")
    else:
        print("FAILURE: Data Asset NOT found in list")

if __name__ == "__main__":
    main()
