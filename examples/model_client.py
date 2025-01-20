import requests
import base64

if __name__ == "__main__":
    import time
    start = time.time()
    for i in range(100):
        with open("./assets/test.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        data = {"image": encoded_string}
        res = requests.post("http://127.0.0.1:8000/classify", json=data)
    end = time.time()

    # 0.169 per image
    print((end - start) / 100)