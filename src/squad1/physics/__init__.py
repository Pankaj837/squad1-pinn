"""Physics engines. ``get_physics(name)`` builds any registered domain's constraint evaluator."""

from __future__ import annotations

from typing import Any

from squad1.errors import PhysicsError
from squad1.physics.base import Physics
from squad1.physics.biot import BiotPhysics
from squad1.physics.darcy import DarcyBC, DarcyPhysics, darcy_residual, solve_pressure
from squad1.physics.domains import PHYSICS_CLASSES


def get_physics(name: str, **kwargs: Any) -> Physics:
    table: dict[str, type[Physics]] = {**PHYSICS_CLASSES, "darcy": DarcyPhysics, "darcy_biot": BiotPhysics}
    if name not in table:
        raise PhysicsError(f"no physics for domain {name!r}; available: {sorted(table)}")
    return table[name](**kwargs)


__all__ = [
    "BiotPhysics",
    "DarcyBC",
    "DarcyPhysics",
    "Physics",
    "darcy_residual",
    "get_physics",
    "solve_pressure",
]
