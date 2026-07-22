"""Legacy job-update smoke script; retained for reference, not current acceptance."""

import json
import urllib.request
import urllib.error
import time

API_URL = "http://127.0.0.1:8000/api"

def make_request(endpoint, method='GET', data=None, token=None):
    url = f"{API_URL}{endpoint}"
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f"Bearer {token}"
        
    req = urllib.request.Request(url, method=method, headers=headers)
    
    if data:
        json_data = json.dumps(data).encode('utf-8')
        req.data = json_data

    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        try:
            return e.code, json.loads(e.read().decode('utf-8'))
        except:
            return e.code, {}
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return None, None

def test_job_update():
    # 0. Login
    print("Logging in...")
    status, login_data = make_request("/auth/login", method='POST', data={
        "username": "admin",
        "password": "admin123"
    })
    
    if status != 200 or not login_data.get("ok"):
        print(f"Login failed: {login_data}")
        return
        
    token = login_data['data']['access_token']
    print("Login successful.")

    # 1. List jobs
    print("Listing jobs...")
    status, data = make_request("/jobs", token=token)
    
    if status != 200 or not data.get("ok"):
        print(f"Error listing jobs: {data}")
        return

    jobs = data.get("data", [])
    target_job = None
    
    if not jobs:
        print("No jobs found. Creating a task and job...")
        # Create task first
        status, task_data = make_request("/tasks", method='POST', data={
            "name": "Test Task",
            "description": "Auto created test task"
        }, token=token)
        
        if status != 200 or not task_data.get("ok"):
             print(f"Failed to create task: {task_data}")
             return
        
        task_id = task_data['data']['id']
        print(f"Created Task ID: {task_id}")
        
        # Create job
        status, job_data = make_request("/jobs", method='POST', data={
            "task_id": task_id,
            "operator_name": "tester",
            "collection_quantity": 10
        }, token=token)
        
        if status != 200 or not job_data.get("ok"):
             print(f"Failed to create job: {job_data}")
             return
             
        target_job = job_data['data']
        print(f"Created new job: {target_job['id']}")
    else:
        # Find a job that is PENDING or RUNNING
        for j in jobs:
            if j['status'] in ['PENDING', 'RUNNING']:
                target_job = j
                break
        
        if not target_job:
            target_job = jobs[0] # Fallback
            print(f"Using existing job: {target_job['id']}")

    job_id = target_job["id"]
    print(f"Testing with Job ID: {job_id}")
    print(f"Current Status: {target_job['status']}")
    print(f"Current Progress: {target_job['progress']}")
    print(f"Current Completed Count: {target_job.get('completed_count')}")

    # 2. Update job
    current_count = (target_job.get("completed_count") or 0) + 1
    total_count = target_job.get("collection_quantity") or 10
    
    payload = {
        "status": "RUNNING",
        "progress": {
            "current": current_count,
            "total": total_count
        }
    }
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    status, patch_data = make_request(f"/jobs/{job_id}", method='PATCH', data=payload, token=token)
    
    if status != 200 or not patch_data.get("ok"):
        print(f"Error updating job: {patch_data}")
        return
        
    updated_job = patch_data.get("data")
    print(f"Updated Job Status: {updated_job['status']}")
    print(f"Updated Job Progress: {updated_job['progress']}")
    print(f"Updated Job Completed Count: {updated_job.get('completed_count')}")
    
    if updated_job['status'] == "RUNNING" and updated_job.get('completed_count') == current_count:
        print("SUCCESS: Job updated correctly via API.")
    else:
        print("FAILURE: Job update returned unexpected values.")

if __name__ == "__main__":
    test_job_update()
