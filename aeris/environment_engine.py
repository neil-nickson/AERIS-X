class EnvironmentEngine:
    def __init__(self, config_manager):
        self.config = config_manager
        
        # Debounce counters for each discrete condition
        self.counters = {
            "LOW-LIGHT": 0,
            "HIGH-VIBRATION": 0,
            "PRECIPITATION": 0,
            "TEMPERATURE-DEVIATION": 0,
            "OUT-OF-VALIDATED-ENVELOPE": 0
        }
        
        # Currently active conditions after debouncing
        self.active_conditions = set()

    def evaluate(self, parsed_data, health_status, dev_data):
        # 1. Check Sensor Health First
        if health_status != "NOMINAL":
            return self._build_result("UNKNOWN", set(), dev_data, health_status)

        if not parsed_data or not dev_data:
            return self._build_result("UNKNOWN", set(), dev_data, health_status)

        # 2. Extract configuration (using fallback defaults for safety)
        cfg = self.config.config.get("thresholds", {})
        low_light_ratio = cfg.get("low_light_ratio", 0.5)
        vib_threshold = cfg.get("vibration_threshold", 15.0)
        temp_dev_thresh = cfg.get("temp_deviation", 5.0)
        env_min_t = cfg.get("prototype_envelope_min_temp", 0.0)
        env_max_t = cfg.get("prototype_envelope_max_temp", 45.0)
        required_samples = cfg.get("stability_samples", 5)

        # 3. Evaluate raw instantaneous conditions
        instant_conditions = set()

        # Light Logic (Directional: only LOW light triggers LOW-LIGHT)
        if dev_data.get("light_ratio", 1.0) < low_light_ratio:
            instant_conditions.add("LOW-LIGHT")

        # Vibration
        if dev_data.get("vibration", 0.0) > vib_threshold:
            instant_conditions.add("HIGH-VIBRATION")

        # Rain
        if parsed_data.get("RAIN_DETECTED", False):
            instant_conditions.add("PRECIPITATION")

        # Temperature Deviation
        if dev_data.get("temp_abs_delta", 0.0) > temp_dev_thresh:
            instant_conditions.add("TEMPERATURE-DEVIATION")

        # Envelope
        curr_temp = parsed_data.get("TEMP", 25.0)
        if curr_temp < env_min_t or curr_temp > env_max_t:
            instant_conditions.add("OUT-OF-VALIDATED-ENVELOPE")

        # 4. Apply Temporal Stability (Debouncing)
        for condition in self.counters.keys():
            if condition in instant_conditions:
                self.counters[condition] += 1
                if self.counters[condition] >= required_samples:
                    self.counters[condition] = required_samples # cap counter
                    self.active_conditions.add(condition)
            else:
                self.counters[condition] -= 1
                if self.counters[condition] <= 0:
                    self.counters[condition] = 0
                    if condition in self.active_conditions:
                        self.active_conditions.remove(condition)

        # 5. Determine Final State
        active = list(self.active_conditions)
        if len(active) == 0:
            final_state = "NORMAL"
        elif len(active) == 1:
            final_state = active[0]
        else:
            final_state = "MULTI-CONDITION"

        return self._build_result(final_state, active, dev_data, health_status)

    def _build_result(self, state, active_conds, dev_data, health):
        return {
            "state": state,
            "active_conditions": sorted(list(active_conds)),
            "health": health,
            "metrics": dev_data
        }
