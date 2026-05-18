from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import cv2
import os
import sys
import tempfile
from skimage.feature import local_binary_pattern

app = FastAPI(
    title="Real/Fake Detection API"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# =====================================================
# GDA CLASS
# =====================================================
class GDA:
    def __init__(self, reg_param=0.0, store_covariance=True):
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None
        self.sigma_inv = None
        self.log_det_sigma = None

    def predict_proba(self, X):
        delta0 = X - self.mu0
        log_p_x_given_y0 = (
            -0.5 * np.sum((delta0 @ self.sigma_inv) * delta0, axis=1)
            - 0.5 * self.log_det_sigma
        )
        log_prob0 = log_p_x_given_y0 + np.log(1 - self.phi)

        delta1 = X - self.mu1
        log_p_x_given_y1 = (
            -0.5 * np.sum((delta1 @ self.sigma_inv) * delta1, axis=1)
            - 0.5 * self.log_det_sigma
        )
        log_prob1 = log_p_x_given_y1 + np.log(self.phi)

        max_log_prob = np.maximum(log_prob0, log_prob1)
        prob0 = np.exp(log_prob0 - max_log_prob)
        prob1 = np.exp(log_prob1 - max_log_prob)

        total = prob0 + prob1

        return np.vstack([
            prob0 / total,
            prob1 / total
        ]).T


setattr(sys.modules['__main__'], 'GDA', GDA)

# =====================================================
# LOAD MODELS
# =====================================================
pipeline = joblib.load("model.pkl")
gda_model = pipeline["model"]
scaler = pipeline["scaler"]
pca = pipeline["pca"]
cfg = pipeline["config"]
K_COMPONENTS = cfg["k_components"]

print("GDA loaded")

lr_data = joblib.load("logistic_regression_model.pkl")
lr_model = lr_data["model"]
lr_pca_scaler = lr_data["pca_output_scaler"]
LR_K_COMPONENTS = lr_data["best_params"]["k_components"]

print("LR loaded")


# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_single_image_features(img_path, cfg):
    img_h, img_w = cfg["img_size"]

    img = cv2.imread(img_path)

    if img is None:
        return None

    img = cv2.resize(img, (img_w, img_h))

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img_gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # RAW
    img_raw_small = cv2.resize(
        img_rgb,
        (128, 128)
    )

    raw_features = (
        img_raw_small / 255.0
    ).flatten()

    # ELA
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        cfg["ela_quality"]
    ]

    _, encoded_img = cv2.imencode(
        ".jpg",
        img,
        encode_param
    )

    recompressed = cv2.imdecode(
        encoded_img,
        1
    )

    recompressed_rgb = cv2.cvtColor(
        recompressed,
        cv2.COLOR_BGR2RGB
    )

    diff = np.abs(
        img_rgb.astype(np.int16)
        - recompressed_rgb.astype(np.int16)
    )

    ela_features = np.concatenate([
        np.mean(diff, axis=(0,1)),
        np.max(diff, axis=(0,1)),
        np.std(diff, axis=(0,1))
    ])

    # FFT
    cy, cx = img_h // 2, img_w // 2

    Y, X = np.ogrid[:img_h, :img_w]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)

    masks = [
        dist <= cfg["fft_low_radius"],
        (dist > cfg["fft_low_radius"]) &
        (dist <= cfg["fft_mid_radius"]),
        dist > cfg["fft_mid_radius"]
    ]

    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)

    magnitude = np.log(
        1 + np.abs(f_shift)
    )

    fft_features = []

    for mask in masks:
        vals = magnitude[mask]
        fft_features.extend([
            np.mean(vals),
            np.std(vals),
            np.max(vals)
        ])

    fft_features = np.array(fft_features)

    # LBP
    lbp = local_binary_pattern(
        img_gray,
        P=cfg["lbp_points"],
        R=cfg["lbp_radius"],
        method=cfg["lbp_method"]
    )

    num_bins = cfg["lbp_points"] + 2

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=num_bins,
        range=(0, num_bins),
        density=True
    )

    # YCbCr
    img_ycbcr = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2YCrCb
    )

    cr_hist, _ = np.histogram(
        img_ycbcr[:,:,1].ravel(),
        bins=32,
        range=(0,256),
        density=True
    )

    cb_hist, _ = np.histogram(
        img_ycbcr[:,:,2].ravel(),
        bins=32,
        range=(0,256),
        density=True
    )

    color_features = np.concatenate([
        cr_hist,
        cb_hist
    ])

    return np.concatenate([
        raw_features,
        ela_features,
        fft_features,
        lbp_hist,
        color_features
    ]).astype(np.float32)


# =====================================================
# API
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Real/Fake Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:
            return {
                "error": "Invalid image"
            }

        temp_path = tempfile.mktemp(
            suffix=".jpg"
        )

        cv2.imwrite(temp_path, img)

        features = extract_single_image_features(
            temp_path,
            cfg
        )

        os.remove(temp_path)

        features = features.reshape(1, -1)

        feat_scaled = scaler.transform(features)
        feat_pca = pca.transform(feat_scaled)

        # GDA
        feat_gda = feat_pca[:, :K_COMPONENTS]
        probs_gda = gda_model.predict_proba(
            feat_gda
        )[0]

        # LR
        feat_lr = feat_pca[:, :LR_K_COMPONENTS]
        feat_lr_scaled = lr_pca_scaler.transform(
            feat_lr
        )

        probs_lr = lr_model.predict_proba(
            feat_lr_scaled
        )[0]

        return {
            "gda": {
                "real": float(probs_gda[0]),
                "fake": float(probs_gda[1])
            },
            "logistic_regression": {
                "real": float(probs_lr[0]),
                "fake": float(probs_lr[1])
            }
        }

    except Exception as e:
        return {
            "error": str(e)
        }