from locust import HttpUser, task, between
import random



class ApiLoadRunner(HttpUser):
    """
    Sends a POST request to the '/predict' endpoint with a JSON payload.

    Args:
        self: The instance of the class.

    Returns:
        None
    """
    wait_time = between(0.5, 2.5)

    @task
    def request(self):
        headers = {
            "Content-Type": "application/json"
        }
        request_body = {
            "accelerations": random.uniform(-1, 1),
            "fetal_movement": random.uniform(-1, 1),
            "uterine_contractions": random.uniform(-1, 1),
            "severe_decelerations": random.uniform(-1, 1)
        }
        self.client.post('/predict', json=request_body, headers=headers)