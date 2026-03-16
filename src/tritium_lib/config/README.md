# Config — Base Settings

Pydantic base settings class for Tritium service configuration. Provides a standardized way to load settings from environment variables, `.env` files, and defaults.

## Usage

```python
from tritium_lib.config import TritiumBaseSettings

class MyServiceSettings(TritiumBaseSettings):
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    site_id: str = "home"
```

Settings are loaded from environment variables (prefixed with `TRITIUM_`) and `.env` files.
