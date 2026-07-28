"""Stable domain vocabulary for the production document catalogue."""

from __future__ import annotations

from enum import StrEnum


class KnowledgeDomain(StrEnum):
    """Technical domains used for routing and source boosting."""

    PHOSPHORIC_ACID_PROCESS = "phosphoric_acid_process"
    PLANT_SPECIFIC = "plant_specific"
    THERMODYNAMICS = "thermodynamics"
    HEAT_TRANSFER = "heat_transfer"
    MASS_TRANSFER = "mass_transfer"
    FLUID_MECHANICS = "fluid_mechanics"
    CRYSTALLIZATION = "crystallization"
    PROCESS_CONTROL = "process_control"
    MPC = "mpc"
    INSTRUMENTATION = "instrumentation"
    EQUIPMENT = "equipment"
    SAFETY = "safety"
    GENERAL_CHEMICAL_ENGINEERING = "general_chemical_engineering"


DOMAIN_DESCRIPTIONS: dict[KnowledgeDomain, str] = {
    KnowledgeDomain.PHOSPHORIC_ACID_PROCESS: (
        "Wet-process phosphoric acid, phosphate reaction, filtration, "
        "concentration and gypsum."
    ),
    KnowledgeDomain.PLANT_SPECIFIC: (
        "Installed plant, equipment, operating conditions and plant sequence."
    ),
    KnowledgeDomain.THERMODYNAMICS: (
        "Properties, phase equilibria, enthalpy, entropy and vapor pressure."
    ),
    KnowledgeDomain.HEAT_TRANSFER: (
        "Conduction, convection, boiling, condensation and heat exchangers."
    ),
    KnowledgeDomain.MASS_TRANSFER: (
        "Diffusion, mass-transfer coefficients and interphase transfer."
    ),
    KnowledgeDomain.FLUID_MECHANICS: (
        "Momentum transport, pressure drop, boundary layers and pipe flow."
    ),
    KnowledgeDomain.CRYSTALLIZATION: (
        "Supersaturation, nucleation, crystal growth and precipitation."
    ),
    KnowledgeDomain.PROCESS_CONTROL: (
        "Dynamics, feedback, PID, multivariable and plantwide control."
    ),
    KnowledgeDomain.MPC: "Model predictive control and optimization.",
    KnowledgeDomain.INSTRUMENTATION: (
        "Industrial measurements, sensors, actuators and control hardware."
    ),
    KnowledgeDomain.EQUIPMENT: (
        "Chemical-process equipment, design, operation and materials."
    ),
    KnowledgeDomain.SAFETY: "Process hazards and safe operation.",
    KnowledgeDomain.GENERAL_CHEMICAL_ENGINEERING: (
        "Cross-domain chemical-engineering reference material."
    ),
}
