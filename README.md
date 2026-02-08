# KYC API for Pakistani CNIC — Advanced Card Processing

A Django REST API that extracts identity data from photographs of Pakistani
**Computerised National Identity Cards (CNICs)**.  
Upload the **front** and **back** images → receive a structured JSON response
containing the cardholder's name, guardian name, CNIC number, date of birth,
issue date, and expiry date.

---

## Features

| Capability | Detail |
|---|---|
| **Card detection** | Custom YOLOv8 ONNX model locates the card region in each image |
| **Face validation** | OpenCV Haar cascade ensures a valid face is present |
| **OCR** | EasyOCR reads English text from the front side |
| **QR decode** | `pyzbar` reads the QR code on the back to cross-validate the CNIC number |
| **Consistent JSON** | Every response uses the same `{ success, message, data, errors }` envelope |

---

## Project Structure

```
├── Django Api/                 # Django project root
│   ├── api/                    # Project settings & root URL config
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── detector/               # Main app
│   │   ├── views.py            # Thin API view (POST endpoint)
│   │   ├── urls.py
│   │   └── services/           # Business logic layer
│   │       ├── image_utils.py  # Image read / grayscale helpers
│   │       ├── model_loader.py # Singleton YOLOv8 & EasyOCR loaders
│   │       └── ocr_service.py  # All OCR, QR, merge & validation logic
│   └── manage.py
├── Model/
│   └── best.onnx               # Trained YOLOv8 ONNX weights
├── yolov8/                     # YOLOv8 ONNX inference wrapper
│   ├── YOLOv8.py
│   └── utils.py
├── Requirements.txt
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **zbar** shared library (required by `pyzbar`)
  - Windows: `pip install pyzbar` bundles DLLs automatically
  - macOS: `brew install zbar`
  - Ubuntu/Debian: `sudo apt install libzbar0`
- The ONNX model file at `Model/best.onnx`

---

## Quick Start

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/<your-user>/KYC-API-for-Pakistani-CNIC-with-Advanced-Card-Processing.git
cd KYC-API-for-Pakistani-CNIC-with-Advanced-Card-Processing

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r Requirements.txt
```

> **Note:** `easyocr` pulls PyTorch as a dependency. For a smaller install on
> CPU-only machines, install the CPU-only PyTorch wheel first — see
> <https://pytorch.org/get-started/locally/>.

### 3. Run database migrations

```bash
cd "Django Api"
python manage.py migrate
```

### 4. Start the development server

```bash
python manage.py runserver
```

The API is now available at **`http://127.0.0.1:8000/`**.

---

## API Reference

### `POST /api/v1/card/detect/`

Upload the front and back images of a Pakistani CNIC.

| Form field | Type | Required | Description |
|---|---|---|---|
| `front_image` | file | Yes | Front side photo (JPEG / PNG) |
| `back_image` | file | Yes | Back side photo (JPEG / PNG) |

#### Example — cURL

```bash
curl -X POST http://127.0.0.1:8000/api/v1/card/detect/ \
  -F "front_image=@front.jpg" \
  -F "back_image=@back.jpg"
```

#### Success response (`200 OK`)

```json
{
  "success": true,
  "message": "ID card processed successfully",
  "data": {
    "card_info": {
      "Name": "MUHAMMAD ALI",
      "Guardian Name": "AHMED ALI",
      "Id Card Number": "3520112345671",
      "Date Of Birth": "010190",
      "Date Of Issue": "010120",
      "Date Of Expiry": "010130"
    },
    "front_cnic": "3520112345671",
    "back_cnic": "3520112345671",
    "ocr_upper_raw": ["MUHAMMAD ALI", "AHMED ALI"],
    "ocr_lower_raw": ["3520112345671", "01.01.1990", "01.01.2020", "01.01.2030"]
  },
  "errors": []
}
```

#### Error responses

| Status | Cause |
|---|---|
| `400` | Missing files or unreadable images |
| `406` | Front & back CNIC numbers do not match |
| `422` | No face detected on the front image |
| `500` | Unexpected server error |

All error responses follow the same envelope:

```json
{
  "success": false,
  "message": "...",
  "data": null,
  "errors": ["error_code"]
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ONNX_MODEL_PATH` | `../Model/best.onnx` (relative to Django project) | Override the path to the ONNX model file |
| `DJANGO_DEBUG` | `True` (in settings) | Set to `False` in production |

---

## Tech Stack

- **Django 4.2** + **Django REST Framework** — API layer
- **YOLOv8 (ONNX)** — card region detection
- **EasyOCR** — optical character recognition
- **pyzbar** — QR code decoding
- **OpenCV** — image processing & face detection
- **django-cors-headers** — cross-origin request support

---

## License

See [LICENSE.txt](LICENSE.txt) for details.
