from core._types import Type, Callable, Generic, T

class Registry(Generic[T]):
    """
    A class to manage and instantiate registered objects.
    """
    def __init__(self) -> None:
        """Initializes the registry."""
        self._registry: dict[str, Type[T]] = {}

    def register(
            self,
            name: str
    ) -> Callable[[Type[T]], Type[T]]:
        """
        A decorator to register a new object.

        Args:
            name (str): The name to register the object under.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self.add(name, cls)
            return cls
        return decorator

    def add(
            self,
            name: str,
            cls: Type[T]
    ) -> None:
        """
        Registers an object in an imperative way.

        This provides a non-decorator way to add an object to the registry.

        Args:
            name (str): The name to register the object under.
            cls (Type[T]): The class to be registered.
        """
        name_lower = name.lower()
        if name_lower in self._registry:
            print(f"Warning: Object '{name}' is already registered. Overwriting.")
        self._registry[name_lower] = cls

    def get(
            self,
            name: str,
            **kwargs
    ) -> T:
        """
        Gets and instantiates an object from the registry.

        Args:
            name (str): The name of the object.
            **kwargs: Arbitrary keyword arguments to pass to the object's constructor.

        Returns:
            An instance of the requested object.

        Raises:
            ValueError: If the object name is not found in the registry.
        """
        name_lower = name.lower()
        if name_lower not in self._registry:
            available = ", ".join(self.registered_objects())
            raise ValueError(
                f"Error: object '{name}' not found. "
                f"Available objects: [{available}]"
            )
        
        # Retrieve the class from the registry
        model_class = self._registry[name_lower]
        
        # Instantiate and return the object
        return model_class(**kwargs)

    def registered_objects(self) -> list[str]:
        """Returns a list of registered object names."""
        return list(self._registry.keys())
