import hashlib
import logging
import sys
import types
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from hyperforge.feature_flag import Features, has_feature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hyperforge.agent import Agent, AgentConfig
    from hyperforge.driver import Driver, DriverConfig
ResolvableType = TypeVar("ResolvableType", types.ModuleType, types.FunctionType, type)
T = TypeVar("T", bound="BaseRegistry")


@dataclass
class BaseRegistry:
    id: str


@dataclass
class AgentRegistry(BaseRegistry):
    agent_type: str
    title: str
    description: str
    config_schema: Type["AgentConfig"]
    klass: Optional[Type["Agent"]] = None


@dataclass
class DriverRegistry(BaseRegistry):
    title: str
    description: str
    config_schema: Type["DriverConfig"]
    klass: Optional[Type["Driver"]] = None


@dataclass
class Registration:
    klass: Any
    config: BaseRegistry


ConfigurationType = List[Tuple[str, Registration]]

_registered_configurations: ConfigurationType = []
# stored as tuple of (type, configuration) so we get keep it in the order
# it is registered even if you mix types of registrations

_registered_configuration_handlers = {}


@dataclass
class Registry:
    agents: Dict[str, AgentRegistry] = field(default_factory=dict)
    drivers: Dict[str, DriverRegistry] = field(default_factory=dict)
    preprocess_agents: Dict[str, AgentRegistry] = field(default_factory=dict)
    postprocess_agents: Dict[str, AgentRegistry] = field(default_factory=dict)
    generation_agents: Dict[str, AgentRegistry] = field(default_factory=dict)
    context_agents: Dict[str, AgentRegistry] = field(default_factory=dict)

    def clear(self):
        self.agents.clear()
        self.drivers.clear()
        self.preprocess_agents.clear()
        self.postprocess_agents.clear()
        self.generation_agents.clear()
        self.context_agents.clear()


GLOBAL_REGISTRY = Registry()


def resolve_dotted_name(name: Any) -> Any:
    """
    import the provided dotted name

    :param name: dotted name
    """
    if not isinstance(name, str):
        return name  # already an object
    names = name.split(".")
    used = names.pop(0)
    found = __import__(used)
    for n in names:
        used += "." + n
        try:
            found = getattr(found, n)
        except AttributeError:
            __import__(used)
            found = getattr(found, n)

    return found


def register_configuration_handler(type_, handler):
    _registered_configuration_handlers[type_] = handler


def register_configuration(klass: ResolvableType, config: BaseRegistry, type_: str):
    value = (type_, Registration(klass=klass, config=config))
    if value not in _registered_configurations:
        # do not register twice
        _registered_configurations.append(value)


def get_caller_module(
    level: int = 2, sys: types.ModuleType = sys
) -> Optional[types.ModuleType]:  # pylint: disable=W0621
    """
    Pulled out of pyramid
    """
    module_globals = sys._getframe(level).f_globals
    module_name = module_globals.get("__name__") or "__main__"
    module = sys.modules[module_name]
    return module


def resolve_module_path(path: str) -> str:
    if len(path) > 0 and path[0] == ".":
        caller_mod = get_caller_module()
        caller_path = get_module_dotted_name(caller_mod)
        caller_path = ".".join(caller_path.split(".")[: -path.count("..")])  # type: ignore
        path = caller_path + "." + path.split("..")[-1].strip(".")
    return path


def get_module_dotted_name(ob) -> Optional[str]:
    return getattr(ob, "__module__", None) or getattr(ob, "__name__", None)


class _base_decorator(Generic[T]):
    configuration_type: str = "base"
    config_klass: Type[T] = BaseRegistry  # type: ignore

    def __init__(self, **config):
        self.config = config

    def __call__(self, klass):
        config_klass_instance = self.config_klass(**self.config)
        register_configuration(klass, config_klass_instance, self.configuration_type)
        return klass


class agent(_base_decorator[AgentRegistry]):
    configuration_type = "agent"
    config_klass = AgentRegistry


class driver(_base_decorator[DriverRegistry]):
    configuration_type = "driver"
    config_klass = DriverRegistry


def scan(path: str):
    """
    Load a module dotted name.

    :param path: dotted name
    """
    path = resolve_module_path(path)
    __import__(path)


def clear():
    _registered_configurations[:] = []


def get_configurations(module_name, type_=None, excluded=None):
    results = []
    for reg_type, registration in _registered_configurations:
        if type_ is not None and reg_type != type_:
            continue
        module = registration.klass
        normalized_name = get_module_dotted_name(resolve_dotted_name(module))

        if normalized_name is not None and (normalized_name + ".").startswith(
            module_name + "."
        ):
            valid = True
            for excluded_module in excluded or []:
                if (normalized_name + ".").startswith(excluded_module + "."):
                    valid = False
                    break
            if valid:
                results.append((reg_type, registration))
    return results


def load_all_configurations(
    module_name, _context: Registry = GLOBAL_REGISTRY, excluded=None
):
    configurations = get_configurations(module_name, excluded=excluded)
    registration: Registration
    for type_, registration in configurations:
        try:
            _registered_configuration_handlers[type_](
                registration.config, registration.klass, _context
            )
        except TypeError:
            logger.error("Can not find %s module" % registration.klass)
            raise
    return configurations


def load_agent(
    registration: AgentRegistry,
    klass: ResolvableType,
    _context: Registry = GLOBAL_REGISTRY,
) -> Any:
    agent_id = registration.id
    registration.klass = resolve_dotted_name(klass)
    if agent_id is None:
        raise Exception("Agent configuration must have an 'id' field")
    if agent_id in _context.agents:
        # Already registered
        return

    _context.agents[agent_id] = registration
    if registration.agent_type == "preprocess":
        _context.preprocess_agents[agent_id] = registration
    elif registration.agent_type == "postprocess":
        _context.postprocess_agents[agent_id] = registration
    elif registration.agent_type == "generation":
        _context.generation_agents[agent_id] = registration
    elif registration.agent_type == "context":
        _context.context_agents[agent_id] = registration


register_configuration_handler("agent", load_agent)


def load_driver(
    registration: DriverRegistry,
    klass: ResolvableType,
    _context: Registry = GLOBAL_REGISTRY,
) -> Any:
    driver_id = registration.id
    registration.klass = resolve_dotted_name(klass)
    if driver_id is None:
        raise Exception("Source configuration must have an 'id' field")
    if driver_id in _context.drivers:
        # Already registered
        return

    _context.drivers[driver_id] = registration


register_configuration_handler("driver", load_driver)


def get_agent_config_instance(
    agent_config: Dict[str, Any], agent_type: str, _context: Registry = GLOBAL_REGISTRY
) -> "AgentConfig":
    module = agent_config.get("module")
    if module is None:
        raise Exception("Agent configuration must have a 'module' field")
    agent_config_klass = get_agent_config_klass(
        module=module, agent_type=agent_type, _context=_context
    )
    return agent_config_klass.model_validate(agent_config)


async def create_agent_instance(
    agent_config: Dict[str, Any], agent_type: str, _config: Registry = GLOBAL_REGISTRY
) -> "Agent":
    module = agent_config.get("module")
    handler = None
    if module is None:
        raise Exception("Agent configuration must have a 'module' field")
    if agent_type == "generation":
        generation_agent = _config.generation_agents.get(module)
        if generation_agent is None:
            raise Exception(
                f"Generation agent module '{module}' is not registered in the generation agents registry"
            )
        config_object = generation_agent.config_schema.model_validate(agent_config)
        handler = generation_agent.klass
    elif agent_type == "preprocess":
        preprocess_agent = _config.preprocess_agents.get(module)
        if preprocess_agent is None:
            raise Exception(
                f"Preprocess agent module '{module}' is not registered in the preprocess agents registry"
            )
        config_object = preprocess_agent.config_schema.model_validate(agent_config)
        handler = preprocess_agent.klass
    elif agent_type == "postprocess":
        postprocess_agent = _config.postprocess_agents.get(module)
        if postprocess_agent is None:
            raise Exception(
                f"Postprocess agent module '{module}' is not registered in the postprocess agents registry"
            )
        config_object = postprocess_agent.config_schema.model_validate(agent_config)
        handler = postprocess_agent.klass
    elif agent_type == "context":
        context_agent = _config.context_agents.get(module)
        if context_agent is None:
            raise Exception(
                f"Context agent module '{module}' is not registered in the context agents registry"
            )
        config_object = context_agent.config_schema.model_validate(agent_config)
        handler = context_agent.klass
    else:
        raise
    if handler is None:
        raise Exception(f"Klass not found for agent module '{module}'")
    return await handler.from_config(config_object)


def get_agent_config_klass(
    module: str, agent_type: Optional[str] = None, _context: Registry = GLOBAL_REGISTRY
) -> Type["AgentConfig"]:
    if agent_type is not None and agent_type != _context.agents[module].agent_type:
        raise Exception(
            f"Agent module '{module}' is registered as type '{_context.agents[module].agent_type}', not '{agent_type}'"
        )
    return _context.agents[module].config_schema


def get_agent_klass(module: str, _context: Registry = GLOBAL_REGISTRY) -> Type["Agent"]:
    agent = _context.agents[module]
    if agent.klass is None:
        raise Exception(f"Klass not found for agent module '{module}'")

    return agent.klass


def validate_driver(item: Any, _context: Registry = GLOBAL_REGISTRY):
    if not isinstance(item, dict):
        raise ValueError("Source configuration must be a dictionary")

    provider = item.get("provider")
    if provider is None:
        raise ValueError("Source configuration must have a 'provider' field")

    if provider not in _context.drivers:
        raise ValueError(
            f"Source module '{provider}' is not registered in the drivers registry"
        )
    return _context.drivers[provider].config_schema.model_validate(item)


def get_driver_config_klass(
    provider: str, _context: Registry = GLOBAL_REGISTRY
) -> Type["DriverConfig"]:
    driver = _context.drivers.get(provider)
    if driver is None:
        raise Exception(f"Source module '{provider}' is not registered")

    return driver.config_schema


def get_driver_klass(
    provider: str, _context: Registry = GLOBAL_REGISTRY
) -> Type["Driver"]:
    driver = _context.drivers[provider]
    if driver.klass is None:
        raise Exception(f"Klass not found for driver module '{provider}'")

    return driver.klass


def get_driver_config_instance(
    driver_config: Dict[str, Any], _context: Registry = GLOBAL_REGISTRY
) -> "DriverConfig":
    provider = driver_config.get("provider")
    if provider is None:
        raise Exception("Driver configuration must have a 'provider' field")
    driver_config_klass = get_driver_config_klass(provider=provider, _context=_context)
    return driver_config_klass.model_validate(driver_config)


def validate_agent_generation(item: Any, _context: Registry = GLOBAL_REGISTRY):
    if not isinstance(item, dict):
        raise ValueError("Generation agent configuration must be a dictionary")

    module = item.get("module")
    if module is None:
        raise ValueError("Generation agent configuration must have a 'module' field")

    if module not in _context.generation_agents:
        raise ValueError(
            f"Generation agent module '{module}' is not registered in the generation agents registry"
        )
    return _context.generation_agents[module].config_schema.model_validate(item)


def validate_agent_preprocess(item: Any, _context: Registry = GLOBAL_REGISTRY):
    if not isinstance(item, dict):
        raise ValueError("Preprocess agent configuration must be a dictionary")

    module = item.get("module")
    if module is None:
        raise ValueError("Preprocess agent configuration must have a 'module' field")

    if module not in _context.preprocess_agents:
        raise ValueError(
            f"Preprocess agent module '{module}' is not registered in the preprocess agents registry"
        )
    return _context.preprocess_agents[module].config_schema.model_validate(item)


def validate_agent_postprocess(item: Any, _context: Registry = GLOBAL_REGISTRY):
    if not isinstance(item, dict):
        raise ValueError("Postprocess agent configuration must be a dictionary")

    module = item.get("module")
    if module is None:
        raise ValueError("Postprocess agent configuration must have a 'module' field")

    if module not in _context.postprocess_agents:
        raise ValueError(
            f"Postprocess agent module '{module}' is not registered in the postprocess agents registry"
        )
    return _context.postprocess_agents[module].config_schema.model_validate(item)


def validate_agent_context(item: Any, _context: Registry = GLOBAL_REGISTRY):
    if not isinstance(item, dict):
        raise ValueError("Context agent configuration must be a dictionary")

    module = item.get("module")
    if module is None:
        raise ValueError("Context agent configuration must have a 'module' field")

    if module not in _context.context_agents:
        raise ValueError(
            f"Context agent module '{module}' is not registered in the context agents registry"
        )
    return _context.context_agents[module].config_schema.model_validate(item)


def enabled_driver(
    driver: DriverRegistry, running_environment: str, account_md5: str
) -> bool:
    return not has_feature(
        flag_key=Features.FILTERED_DRIVERS_FEATURE_FLAG.format(
            environment=running_environment
        ),
        context={"drivers": driver.id},
    ) or has_feature(
        flag_key=Features.RAO_ACCOUNT_ENABLED_FEATURE_FLAG.format(
            account_md5=account_md5
        ),
        context={"drivers": driver.id},
        default=False,
    )


def create_driver_schema(driver: DriverRegistry) -> Dict[str, Any]:
    return {
        "id": driver.id,
        "title": driver.title,
        "description": driver.description,
        "config_schema": driver.config_schema.model_json_schema(),
    }


def get_driver_agent_schemas(
    running_environment: str = "production",
    account_id: str = "account_id",
    _context: Registry = GLOBAL_REGISTRY,
) -> Dict[str, Dict[str, Any]]:
    return {
        id: create_driver_schema(driver)
        for id, driver in _context.drivers.items()
        if enabled_driver(driver, running_environment, account_id)
    }


def enabled_agent(
    agent: AgentRegistry, running_environment: str, account_md5: str
) -> bool:
    """Filters out agents and drivers from the schema based on feature flags.
    Returns lists of filtered agents and drivers.

    If the driver or agent is enabled at account-level, we don't filter it out even if the environment-level flag to disable it is enabled.
    This allows us to have a global kill switch for agents and drivers, while still allowing specific accounts to have access to them if needed.

    """

    return not has_feature(
        flag_key=Features.FILTERED_AGENTS_FEATURE_FLAG.format(
            environment=running_environment
        ),
        context={"agents": agent.id},
    ) or has_feature(
        flag_key=Features.RAO_ACCOUNT_ENABLED_FEATURE_FLAG.format(
            account_md5=account_md5
        ),
        context={"agents": agent.id},
        default=False,
    )


def create_agent_schema(agent: AgentRegistry) -> Dict[str, Any]:
    published_functions = (
        agent.klass.__published_functions__ if agent.klass is not None else {}
    )
    return {
        "id": agent.id,
        "agent_type": agent.agent_type,
        "title": agent.title,
        "description": agent.description,
        "config_schema": agent.config_schema.model_json_schema(),
        "functions": {
            function_id: definition.description
            for function_id, definition in published_functions.items()
        },
    }


def get_context_agent_schemas(
    running_environment: str = "production",
    account_id: str = "account_id",
    _context: Registry = GLOBAL_REGISTRY,
) -> Dict[str, Dict[str, Any]]:
    account_md5 = hashlib.md5(account_id.encode()).hexdigest()

    return {
        id: create_agent_schema(agent)
        for id, agent in _context.context_agents.items()
        if enabled_agent(agent, running_environment, account_md5)
    }


def get_preprocess_agent_schemas(
    running_environment: str = "production",
    account_id: str = "account_id",
    _context: Registry = GLOBAL_REGISTRY,
) -> Dict[str, Dict[str, Any]]:
    account_md5 = hashlib.md5(account_id.encode()).hexdigest()
    return {
        id: create_agent_schema(agent)
        for id, agent in _context.preprocess_agents.items()
        if enabled_agent(agent, running_environment, account_md5)
    }


def get_postprocess_agent_schemas(
    running_environment: str = "production",
    account_id: str = "account_id",
    _context: Registry = GLOBAL_REGISTRY,
) -> Dict[str, Dict[str, Any]]:
    account_md5 = hashlib.md5(account_id.encode()).hexdigest()

    return {
        id: create_agent_schema(agent)
        for id, agent in _context.postprocess_agents.items()
        if enabled_agent(agent, running_environment, account_md5)
    }


def get_generation_agent_schemas(
    running_environment: str = "production",
    account_id: str = "account_id",
    _context: Registry = GLOBAL_REGISTRY,
) -> Dict[str, Dict[str, Any]]:
    account_md5 = hashlib.md5(account_id.encode()).hexdigest()

    return {
        id: create_agent_schema(agent)
        for id, agent in _context.generation_agents.items()
        if enabled_agent(agent, running_environment, account_md5)
    }
