#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import Any

from ..MlirConfig import MlirConfig
from ..MlirProgram import RawMlirProgram

__all__ = ["MlirTarget"]


class MlirTarget(ABC):
    """An abstract Mlir Target for the Mlir Compiler

    A Mlir Target is responsible for the final steps of the lowering and code generation
    for a specific Target. A Target can be the software backend dedicated to a Hardware,      or a new code generation for an already known Hardware.
    """

    def __init__(self, config: MlirConfig):
        self._config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, new_config: MlirConfig):
        self._config = new_config

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def arch(self) -> str: ...

    @abstractmethod
    def generate_code_for_target(
        self, mlir_program: RawMlirProgram, **kwargs: Any
    ) -> None:
        """
        Generate the code and produce the shared lib or executable for the target.
        The mlir_program represents the IR after XTC transformations.
        """
        ...
