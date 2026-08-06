"""Aggregates every v1 endpoint router into a single APIRouter."""

from fastapi import APIRouter

from app.api.v1.endpoints import dewpoint, fire, fire_nrt, health, ml, temperature

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(temperature.router)
api_router.include_router(dewpoint.router)
api_router.include_router(fire.router)
api_router.include_router(fire_nrt.router)
api_router.include_router(ml.router)
