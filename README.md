# APIGen-MT: Multi-Turn API Data Generation Pipeline

APIGen-MT is a synthetic data generation pipeline for creating high-quality, multi-turn conversational AI datasets grounded in executable API calls. It uses a two-phase process to first create verifiable "blueprints" of tasks and then simulate realistic user-agent conversations.

## Project Structure

```text
/
├───apigen_mt/      # Core Python source code for the pipeline
│   ├───pipeline/   # Main logic for Phase 1 (blueprinting) and Phase 2 (rollout)
│   ├───env/        # Base environment and tool definitions, with sub-folders for each domain
│   ├───policies/   # Domain-specific policy checks
│   ├───agents/     # Implementations of the Human and Assistant agents for Phase 2
│   └───...         # Other supporting modules (schemas, sampling, etc.)
├───configs/        # YAML configuration files
│   ├───sampling/   # Persona, domain data, and policy configurations
│   └───schemas/    # Tool schemas for each domain
├───data/           # Default output directory for generated data
└───README.md       # This file
```

## Key Features

- **Two-Phase Generation**: Decouples task validation from conversation simulation for higher data quality.
- **Verifiable Blueprints**: All tasks are validated through execution, policy checks, and an LLM review committee.
- **Realistic Conversations**: Simulates natural, multi-turn dialogues with a human-like persona driver.
- **Reverse Task Recombination**: Automatically composes longer, more complex tasks from simpler, validated ones.
- **Extensible Framework**: Easily add new API domains, tools, and policies.
- **YAML-Driven Configuration**: Manage tool schemas and domain data in simple YAML files.

## Quick Start

1. **Prerequisites**: Python 3.10+

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure LLM**: Set your LLM provider credentials as environment variables. The pipeline uses `litellm` for broad compatibility.

    ```bash
    # Example for OpenAI
    export LLM_MODEL=gpt-4o-mini
    export OPENAI_API_KEY="sk-..."
    ```

    To run without an LLM (using offline mock stubs for testing), set `export MOCK_LLM=true`.
    You can also bound request latency via `export LLM_REQUEST_TIMEOUT_SEC=10`.

4. **Run Generation**: Execute the full pipeline to generate 5 blueprints and then roll out trajectories for them.

    ```bash
    # Generate blueprints (Phase 1)
    python -m apigen_mt.cli gen-blueprints --domain retail --count 5

    # Roll out trajectories (Phase 2)
    python -m apigen_mt.cli roll-out --in data/blueprints.jsonl --agent natural
    ```

## How It Works

The pipeline operates in two main phases:

1. **Phase 1: Blueprinting (`gen-blueprints`)**: A planner-first pass now proposes grounded, valid action sequences using the API graph and domain rows (no free-form IDs), and only then uses the LLM to phrase the instruction/outputs. Proposals are validated through execution, policy checks, and optionally an LLM review committee. Only tasks that pass all checks are saved as verified blueprints.

2. **Phase 2: Trajectory Rollout (`roll-out`)**: Each blueprint is used as the goal for a simulated conversation between a `HumanDriver` (the user) and an `AssistantAgent` (the AI). The resulting dialogue and tool interactions are saved as a trajectory, but only if the final outcome matches the blueprint's requirements.

## Usage Details

The primary entry point is the command-line interface (`apigen_mt/cli.py`).

### Generating Blueprints (`gen-blueprints`)

This command runs the Phase 1 blueprinting process.

```bash
python -m apigen_mt.cli gen-blueprints --domain retail --count 10
```

**Controlling Task Complexity:**

- `--recombine`: After generating simple blueprints, this flag triggers the recombination process to create and validate more complex, multi-part tasks.

**Controlling Generation Quality & Acceptance:**

- `--max-attempts <N>`: Number of times to retry generating a valid blueprint (default: 5).
- `--best-of <N>`: For each attempt, sample N candidates; a planner-first set is scored by execution/policy, and LLM proposals are evaluated and the best candidate is selected by minimal steps/writes (default: 3).
- `--skip-committee`: A useful flag for bootstrapping. It accepts any task that passes the execution and policy checks, skipping the more subjective LLM review.
- `--min-correctness <0-5>`: The minimum score required from the review committee (default: 3).

### Rolling Out Trajectories (`roll-out`)

This command runs the Phase 2 conversation simulation using a file of validated blueprints.

```bash
python -m apigen_mt.cli roll-out --in data/blueprints.jsonl
```

**Agent Modes (`--agent`):**

- `replay` (Default): A simple agent that executes the blueprint's tool calls deterministically.
- `natural`: A more advanced simulation where a `HumanDriver` and `AssistantAgent` have a natural, multi-turn conversation involving slot-filling and dynamic function calling.

In natural mode, the assistant only sees tools allowed by the blueprint (plus read-only resolvers), validates with read tools before writes, and stops early as soon as the environment state matches the blueprint diff. Final messages explicitly include the expected outputs to guarantee verifiability.

### Retail Domain Helper Tools

The retail domain now includes two resolver tools to avoid ID guessing:

- `resolve_user_by_name(name: string) -> { user_id }` (read-only)
- `find_delivered_order(user_id: string) -> { orders: string[] }` (read-only)

These are available to the assistant during natural rollouts and are defined in both the environment and `configs/schemas/retail.yaml`.

## How to Add a New Domain

A "domain" is a collection of APIs, state logic, policies, and configurations that represent a specific use case (e.g., `retail`, `airline`, `booking`). Adding a new domain is the primary way to customize the pipeline.

Here is a step-by-step guide to creating a new domain called `booking`.

### Step 1: Plan Your Domain

Before writing code, plan the components:

- **Domain Name**: `booking`
- **State**: What information does the environment need to track? Let's say it's a dictionary of reservations, keyed by a `booking_id`.
- **Tools**: What actions can be taken?
    - `search_hotels(location: str, dates: str)`: A read-only tool.
    - `book_hotel(hotel_id: str, user_id: str)`: A write tool that creates a reservation.
    - `cancel_booking(booking_id: str, user_id: str)`: A write tool that modifies state.
- **Dependencies**: A user must `search_hotels` before they can `book_hotel`.

### Step 2: Bootstrap Configuration Files

Use the `gen-domain-spec` CLI command to have an LLM create the initial YAML configuration files for your tool schemas and sampling data.

```bash
python -m apigen_mt.cli gen-domain-spec --name booking --hint "A hotel booking system with tools to search, book, and cancel."
```

This command creates two files:

- `configs/schemas/booking.yaml`: Defines the schemas for your tools (`search_hotels`, etc.).
- `configs/sampling/booking.yaml`: Provides sample data (`domain_rows`), user personas, and policy descriptions to guide the data generation.

Review these generated files and refine them as needed.

### Step 3: Implement the Environment

Create a new file: `apigen_mt/env/domains/booking_env.py`. This file contains the actual executable Python code for your tools.

```python
# apigen_mt/env/domains/booking_env.py
from apigen_mt.env.base import Environment, Tool
from apigen_mt.specs.schema_yaml import load_tools

class BookingEnv(Environment):
    def __init__(self):
        super().__init__()
        # Internal state for the environment
        self._bookings = {
            "b123": {"user_id": "u456", "hotel_id": "h789", "status": "CONFIRMED"}
        }
        self._next_id = 456

        # Load tool schemas from YAML
        tool_schemas = {t["name"]: t for t in load_tools("booking").get("tools", [])}

        # Register your tools
        self.register_tool(Tool(
            name="search_hotels", 
            func=self._search_hotels,
            schema=tool_schemas["search_hotels"]["schema"],
        ))
        self.register_tool(Tool(
            name="book_hotel", 
            func=self._book_hotel, 
            schema=tool_schemas["book_hotel"]["schema"],
            write=True, 
            deps=("search_hotels",)
        ))
        self.register_tool(Tool(
            name="cancel_booking", 
            func=self._cancel_booking, 
            schema=tool_schemas["cancel_booking"]["schema"],
            write=True
        ))

    # Implement state management methods
    def snapshot(self):
        return {"bookings": self._bookings}

    def restore(self, state):
        self._bookings = state.get("bookings", {})

    # Implement tool functions
    def _search_hotels(self, args):
        return [{"hotel_id": "h789", "name": "Grand Hotel"}, {"hotel_id": "h456", "name": "Plaza"}]

    def _book_hotel(self, args):
        booking_id = f"b{self._next_id}"
        self._next_id += 1
        self._bookings[booking_id] = {"user_id": args["user_id"], "hotel_id": args["hotel_id"], "status": "CONFIRMED"}
        return {"ok": True, "booking_id": booking_id}

    def _cancel_booking(self, args):
        booking = self._bookings.get(args["booking_id"])
        if not booking or booking["user_id"] != args["user_id"]:
            raise ValueError("Booking not found or user mismatch")
        booking["status"] = "CANCELLED"
        return {"ok": True}

```

### Step 4: Implement the Policy Suite

Create a new file: `apigen_mt/policies/booking_policies.py`. This is where you define rules that check the *trace* of an execution for violations.

```python
# apigen_mt/policies/booking_policies.py
from apigen_mt.policies.base import PolicySuite

booking_policy_suite = PolicySuite(name="booking")

@booking_policy_suite.add
def check_user_consistency(trace):
    """Check that the same user_id is used across all actions."""
    user_ids = set()
    for step in trace.get("steps", []):
        uid = step.get("arguments", {}).get("user_id")
        if uid:
            user_ids.add(uid)
    if len(user_ids) > 1:
        return False, f"Inconsistent user_id used across steps: {user_ids}"
    return True, ""

```

### Step 5: Register Your Domain

Finally, make the pipeline aware of your new domain by adding it to the registries.

1. **Register the Environment** in `apigen_mt/env/registry.py`:

    ```python
    # ... existing code
    elif domain == "retail":
        from .domains.retail_env import RetailEnv
        return RetailEnv()
    elif domain == "booking":  # Add this block
        from .domains.booking_env import BookingEnv
        return BookingEnv()
    # ... existing code
    ```

2. **Register the Policy Suite** in `apigen_mt/policies/registry.py`:

    ```python
    # ... existing code
    elif domain == "retail":
        from .retail_policies import retail_policy_suite
        return retail_policy_suite
    elif domain == "booking":  # Add this block
        from .booking_policies import booking_policy_suite
        return booking_policy_suite
    # ... existing code
    ```

### Step 6: Generate Data!

Your new domain is now fully integrated. You can start generating high-quality, validated data for it.

```bash
python -m apigen_mt.cli gen-blueprints --domain booking --count 10
```
