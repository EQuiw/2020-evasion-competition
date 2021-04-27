import os
import unittest
import time

import requests
import json

URL = 'http://127.0.0.1:8080'
HEADERS = {
    'Content-Type': 'application/octet-stream'
}
TEST_OBJ_DIR = './test/test_objects'

MAX_BYTES = 2097152


class TestPostRequest(unittest.TestCase):
    def test_connection(self):
        response = requests.get(URL)
        # expected: HTTP 405 (method not allowed)
        self.assertEqual(response.status_code, 405)

    def test_empty(self):
        bytez = bytearray(0)
        response = requests.post(URL, data=bytez, headers=HEADERS)
        content = json.loads(response.text)
        self.assertTrue(response.ok)
        self.assertEqual(content['result'], 0)

    def test_no_pe_file(self):
        bytez = '\x01\x01\x01\x01'.encode()
        response = requests.post(URL, data=bytez, headers=HEADERS)
        content = json.loads(response.text)
        self.assertTrue(response.ok)
        self.assertEqual(content['result'], 1)

    def test_too_large_file(self):
        bytez = ('\x01'*(MAX_BYTES + 1)).encode()
        response = requests.post(URL, data=bytez, headers=HEADERS)
        content = json.loads(response.text)
        self.assertTrue(response.ok)
        self.assertEqual(content['result'], 1)

    def test_benign(self):
        filename = 'putty.exe'
        with open(os.path.join(TEST_OBJ_DIR, filename), 'rb') as f:
            bytez = f.read()
        response = requests.post(URL, data=bytez, headers=HEADERS)
        content = json.loads(response.text)

        self.assertTrue(response.ok)
        self.assertEqual(content['result'], 0)

    # def test_malicious(self):
    #     filename = 'malicious'
    #     with open(os.path.join(TEST_OBJ_DIR, filename), 'rb') as f:
    #         bytez = f.read()
    #     response = requests.post(URL, data=bytez, headers=HEADERS)
    #     content = json.loads(response.text)
    #
    #     self.assertTrue(response.ok)
    #     self.assertEqual(content['result'], 1)

if __name__ == '__main__':
    unittest.main()
