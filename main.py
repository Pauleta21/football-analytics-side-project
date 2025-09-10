from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import logging
import json
import os

# railway logger
class CustomRailwayLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "level": record.levelname.lower(),
            "message": record.getMessage()
        }
        return json.dumps(log_record)

def get_logger():
    logger = logging.getLogger("railway_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    handler.setFormatter(CustomRailwayLogFormatter())
    logger.addHandler(handler)
    return logger

logger = get_logger()

#load models and dataset 
# using logger to show info about what is happening)
# using try-except block to stop the app if not loading the models
try:
    goals_model = pickle.load(open("goals_model.pkl", "rb"))
    logger.info("Goals model loaded successfully")

    assists_model = pickle.load(open("assists_model.pkl", "rb"))
    logger.info("Assists model loaded successfully")

    similar_model = pickle.load(open("similar_players_model.pkl", "rb"))
    logger.info("Similar players model loaded successfully")

    performance_model = pickle.load(open("performance_model.pkl", "rb"))
    logger.info("Performance model loaded successfully")

    players_df = pd.read_csv("players.csv")
    logger.info(f"File players.csv loaded successfully, shape={players_df.shape}")

except Exception as e:
    logger.error("Error loading models or dataset", exc_info=True)
    raise e  # stop app if models don’t load

#init the app
app = FastAPI()

#INPUT SCHEMAS - PYDANTIC
# for the performance model (new player)
class PerformanceInput(BaseModel):
    xG: float
    xA: float
    prog_passes: float
    prog_carries: float
    minutes_played: float

# for the performance model (based on name)
class NameInput(BaseModel):
    player_name: str

# for similar players (based on name)
class NameOnly(BaseModel):
    player_name: str

# for future goals or assists prediction (new player)
class StatsInput(BaseModel):
    xG: float
    xA: float
    prog_passes: float
    prog_carries: float
    minutes_played: float


#ENDPOINTS 
# using logger to show information about whats happening
# predict future goals (new player)
@app.post("/predict-goals")
def predict_goals(input_data: StatsInput):
    logger.info(f"Predicting goals for input={input_data.dict()}")
    features = [[
        input_data.xG,
        input_data.xA,
        input_data.prog_passes,
        input_data.prog_carries,
        input_data.minutes_played
    ]]
    predicted_goals = goals_model.predict(features)[0]
    return {"predicted_goals": predicted_goals}

# predict future assists (new player)
@app.post("/predict-assists")
def predict_assists(input_data: StatsInput):
    logger.info(f"Predicting assists for input={input_data.dict()}")
    features = [[
        input_data.xG,
        input_data.xA,
        input_data.prog_passes,
        input_data.prog_carries,
        input_data.minutes_played
    ]]
    predicted_assists = assists_model.predict(features)[0]
    return {"predicted_assists": predicted_assists}

# compare player - find similar players
@app.post("/compare-player")
def compare_player(input_data: NameOnly):
    name = input_data.player_name
    logger.info(f"Comparing player={name}")

    if name not in players_df["Player"].values:
        logger.warning(f"Player {name} not found in dataset")
        raise HTTPException(status_code=404, detail="Player not found in dataset")

    row = players_df[players_df["Player"] == name]
    numeric_features = players_df.select_dtypes(include=["number"]).columns
    features = row[numeric_features].values

    cluster_label = similar_model.predict(features)[0]
    similar_players = players_df[similar_model.labels_ == cluster_label]["Player"].tolist()
    similar_players = [p for p in similar_players if p != name]

    return {"player_name": name, "similar_players": similar_players}


# classify new player performance (over/under/well)
@app.post("/rate-player-new")
def rate_new_player(input_data: PerformanceInput):
    logger.info(f"Rating new player input={input_data.dict()}")
    features = [[
        input_data.xG,
        input_data.xA,
        input_data.prog_passes,
        input_data.prog_carries,
        input_data.minutes_played
    ]]
    label = performance_model.predict(features)[0]
    return {"performance_label": label}


# classify existing players performance (by name)
@app.post("/rate-player-existing")
def rate_existing_player(input_data: NameOnly):
    name = input_data.player_name
    logger.info(f"Rating existing player={name}")

    if name not in players_df["Player"].values:
        logger.warning(f"Player {name} not found in dataset")
        raise HTTPException(status_code=404, detail="Player not found in dataset")

    row = players_df[players_df["Player"] == name]
    features = row[["Expected_Goals", "Expected_Assists", "Progressive_Passes", "Progressive_Carries", "Minutes_Played"]].values
    label = performance_model.predict(features)[0]
    return {"player_name": name, "performance_label": label}

# definir o endpoint GET para a raiz da API ('/')
# mensagem de boas-vindas e orienta interagir em /docs
# also using logger to show what happened
@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"message": "Welcome to the Player ML API! Go to /docs to use it."}
