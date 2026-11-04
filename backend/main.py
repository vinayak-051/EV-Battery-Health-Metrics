from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from predictor import predict

app = FastAPI(title="EV Battery Health API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class BatteryInput(BaseModel):
    cycle: float = Field(..., gt=0, description="Discharge cycle number")
    energy_Wh: float = Field(..., gt=0, description="Discharge energy in Wh")
    voltage_mean: float = Field(..., gt=0, description="Mean discharge voltage (V)")
    current_mean: float = Field(..., gt=0, description="Mean discharge current (A)")
    temperature_mean: float = Field(..., description="Mean battery temperature (°C)")


class PredictionResult(BaseModel):
    soh: float
    rul: int
    status: str
    discharge_duration_min: float


@app.post("/predict", response_model=PredictionResult)
def predict_health(data: BatteryInput):
    try:
        result = predict(
            cycle=data.cycle,
            energy_Wh=data.energy_Wh,
            voltage_mean=data.voltage_mean,
            current_mean=data.current_mean,
            temperature_mean=data.temperature_mean,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}
