"""Registry module."""

import logging
from typing import Callable

from futureframe.baselines import create_baseline_pipeline
from futureframe.models.tabtext import TabText, TabTextXGBoostClassifier

log = logging.getLogger(__name__)

predictors_registry = dict(
    TabText=TabText,
    TabTextXGBoostClassifier=TabTextXGBoostClassifier,
)


def register_predictor(predictor_class: type):
    """
    Register a predictor class.

    Args:
        predictor_name: The name of the predictor.
            This name will be used to identify and retrieve the predictor class.
        predictor_class: The class of the predictor.
            This class should implement the necessary methods and functionality for making predictions.

    Returns:
        None

    Example:
        ```python
        >>> from my_module import MyPredictorClass
        >>> register_predictor("my_predictor", MyPredictorClass)
        ```
    """
    predictor_name = predictor_class.__name__
    predictors_registry[predictor_name] = predictor_class


def get_predictor_class_by_name(predictor_name: str):
    """
    Get a predictor class by its registered name.

    Args:
        predictor_name: The name of the predictor that was used during registration.

    Returns:
        type: The class of the predictor or None if the predictor name is not found in the registry.

    Example:
        ```python
        >>> predictor_class = get_predictor_class("my_predictor")
        >>> predictor = predictor_class()  # Create an instance of the predictor class
        ```
    """
    return predictors_registry.get(predictor_name)


def get_predictor_class_by_idx(idx: int):
    """
    Get a predictor class by its registered index.
    Args:
        idx: The index of the predictor that was used during registration.
    Returns:
        type: The class of the predictor or None if the predictor index is not found in the registry.
    Example:
        ```python
        >>> predictor_class = get_predictor_class(0)
        >>> predictor = predictor_class()  # Create an instance of the predictor class
        ```
    """
    return list(predictors_registry.values())[idx]


def register_predictor_decorator() -> Callable:
    """
    A decorator to register a predictor class with a given name.

    Args:
        predictor_name (str): The name to associate with the predictor class.

    Returns:
        Callable: A decorator function that registers the decorated class as a predictor.

    Example:
        ```python
        >>> @register_predictor_decorator
        >>> class MyDecoratedPredictorClass:
        >>> # Implement your predictor class here
        >>>     pass
        ```
    """

    def decorator(predictor_class: type) -> type:
        register_predictor(predictor_class)
        return predictor_class

    return decorator


def create_predictor(
    predictor_name: str, column_names, task_type=None, numeric_features=None, categorical_features=None, **kwargs
):
    """
    Create an instance of a registered predictor class.

    Args:
        predictor_name: The name of the predictor that was used during registration.
        *args: Variable length argument list that will be passed to the predictor class constructor.
        **kwargs: Keyword arguments that will be passed to the predictor class constructor.

    Returns:
        object: An instance of the predictor class.

    Raises:
        ValueError: If the predictor name is not found in the registry.

    Example:
        ```python
        >>> predictor = create_predictor(
        ...     "my_predictor", arg1, arg2, kwarg1="value1", kwarg2="value2"
        ... )
        ```
    """
    if predictor_name not in predictors_registry:
        return create_baseline_pipeline(predictor_name, task_type, numeric_features, categorical_features, **kwargs)

    predictor_class = get_predictor_class_by_name(predictor_name)
    if predictor_class is None:
        raise ValueError(f"Predictor '{predictor_name}' not found in the registry.")
    return predictor_class(**kwargs)