"""
Registry module for managing and instantiating registered objects.

This module provides a generic Registry class that allows registration of classes
by name and their subsequent instantiation with arbitrary keyword arguments.
It supports both decorator-based and imperative registration patterns.
"""
from core._types import Type, Callable, Generic, T

class Registry(Generic[T]):
    """
    A generic registry for managing and instantiating registered objects.
    
    This class provides a type-safe way to register classes by name and retrieve
    instances of those classes with arbitrary initialization parameters. All
    registered names are stored in lowercase for case-insensitive lookup.
    
    Type Parameters:
        T: The base type that all registered classes should conform to.
    
    Attributes:
        _registry (dict[str, Type[T]]): Internal dictionary mapping lowercase names
            to registered class types.
    """
    def __init__(self) -> None:
        """
        Initializes an empty registry.
        
        Creates a new Registry instance with no registered objects.
        """
        self._registry: dict[str, Type[T]] = {}

    def register(
            self,
            name: str
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class in the registry.
        
        This method returns a decorator that can be used to register a class
        under a specified name. The name is automatically converted to lowercase
        for case-insensitive lookup.

        Args:
            name (str): The name to register the class under. Will be converted
                to lowercase internally.

        Returns:
            Callable[[Type[T]], Type[T]]: A decorator function that registers
                the class and returns it unchanged.
        
        Example:
            >>> registry = Registry[MyBaseClass]()
            >>> @registry.register("my_impl")
            ... class MyImplementation(MyBaseClass):
            ...     pass
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
        Registers a class in the registry imperatively.

        This provides a non-decorator way to add a class to the registry.
        The name is automatically converted to lowercase for case-insensitive
        lookup. If the name is already registered, a warning is printed and
        the existing registration is overwritten.

        Args:
            name (str): The name to register the class under. Will be converted
                to lowercase internally.
            cls (Type[T]): The class type to be registered.
        
        Example:
            >>> registry = Registry[MyBaseClass]()
            >>> registry.add("my_impl", MyImplementation)
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
        Retrieves and instantiates a registered class by name.
        
        Looks up the class registered under the given name (case-insensitive)
        and instantiates it with the provided keyword arguments.

        Args:
            name (str): The name of the registered class to instantiate.
                Lookup is case-insensitive.
            **kwargs: Arbitrary keyword arguments to pass to the class constructor.

        Returns:
            T: An instance of the requested registered class.

        Raises:
            ValueError: If the name is not found in the registry. The error
                message includes a list of all available registered names.
        
        Example:
            >>> registry = Registry[MyBaseClass]()
            >>> registry.add("my_impl", MyImplementation)
            >>> instance = registry.get("my_impl", param1=value1, param2=value2)
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
        """
        Returns a list of all registered object names.
        
        Provides a list of all names currently registered in the registry.
        All names are in lowercase as they are stored that way internally.

        Returns:
            list[str]: A list of all registered names (in lowercase).
        
        Example:
            >>> registry = Registry[MyBaseClass]()
            >>> registry.add("impl1", Implementation1)
            >>> registry.add("impl2", Implementation2)
            >>> registry.registered_objects()
            ['impl1', 'impl2']
        """
        return list(self._registry.keys())
