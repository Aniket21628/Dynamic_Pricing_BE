import numpy as np
from scipy.spatial.distance import euclidean

class RideHailingEnv:
    def __init__(self, num_zones=16, num_vehicles=100, time_steps_per_day=1440):
        self.num_zones = num_zones
        self.num_vehicles = num_vehicles
        self.time_steps_per_day = time_steps_per_day
        self.current_step = 0
        
        # Vehicle state: [zone_id, is_idle]
        self.vehicle_states = np.zeros((num_vehicles, 2), dtype=np.int32)
        self.vehicle_states[:, 1] = 1  # Start as idle
        
        # Demand queue by zone
        self.demand_queue = {i: [] for i in range(num_zones)}
        
        # Pricing multipliers per zone
        self.price_multipliers = np.ones(num_zones)
        
        # Zone centroids (grid layout)
        self.zone_positions = self._create_zone_grid()
        
        # Revenue tracking
        self.total_revenue = 0.0
        self.rides_completed = 0
        self.empty_trips = 0
        
    def _create_zone_grid(self):
        """Create grid of zone positions"""
        side = int(np.sqrt(self.num_zones))
        positions = {}
        for i in range(self.num_zones):
            row = i // side
            col = i % side
            positions[i] = np.array([row * 100, col * 100])
        return positions
    
    def reset(self):
        self.current_step = 0
        self.vehicle_states[:, 0] = 0  # optionally reset positions to zone 0
        self.vehicle_states[:, 1] = 1
        self.total_revenue = 0.0
        self.rides_completed = 0
        self.empty_trips = 0
        self.demand_queue = {i: [] for i in range(self.num_zones)}
        return self._get_state()
    
    def _get_state(self):
        """Get current state vector"""
        # Count idle vehicles per zone
        idle_per_zone = np.zeros(self.num_zones)
        for zone in range(self.num_zones):
            idle_per_zone[zone] = np.sum((self.vehicle_states[:, 0] == zone) & (self.vehicle_states[:, 1] == 1))
        
        # Count demand per zone
        demand_per_zone = np.array([len(self.demand_queue[z]) for z in range(self.num_zones)])
        
        state = np.concatenate([
            idle_per_zone,
            demand_per_zone,
            self.price_multipliers,
            [self.current_step / self.time_steps_per_day]
        ])
        
        return state.astype(np.float32)
    
    def step(self, actions):
        """
        Execute one step with pricing actions
        actions: array of price multipliers for each zone (expected positive)
        Returns: (next_state, step_revenue, done, info)
        - step_revenue: revenue earned in this single step (not cumulative)
        - info contains step-level metrics: {'step_revenue':..., 'step_empty_trips':...}
        """
        # Update price multipliers (env expects multipliers, we clip to valid range)
        self.price_multipliers = np.clip(actions, 0.33, 3.0)
        
        # Generate demand based on time of day
        self._generate_demand()
        
        # Match vehicles to rides -> returns step revenue (not cumulative)
        step_revenue = self._match_and_dispatch()
        
        # Vehicle repositioning -> returns number of empty repositioning trips this step
        step_empty_trips = self._reposition_vehicles()
        self.empty_trips += step_empty_trips
        
        # Update time
        self.current_step += 1
        done = self.current_step >= self.time_steps_per_day
        
        info = {
            'step_revenue': step_revenue,
            'step_empty_trips': step_empty_trips,
            'total_revenue': self.total_revenue,
            'rides_completed': self.rides_completed,
            'empty_trips': self.empty_trips
        }
        
        return self._get_state(), step_revenue, done, info
    
    def _generate_demand(self):
        """Generate demand based on time and price"""
        hour = (self.current_step % self.time_steps_per_day) / 60
        
        # Base demand varies by hour
        if 7 <= hour < 10 or 17 <= hour < 20:
            base_demand = 20  # Peak hours
        elif 10 <= hour < 17:
            base_demand = 10  # Midday
        else:
            base_demand = 5   # Night
        
        for zone in range(self.num_zones):
            # Price affects demand
            multiplier = self.price_multipliers[zone]
            demand = max(0, int(base_demand / multiplier + np.random.normal(0, 2)))
            
            for _ in range(demand):
                dest_zone = np.random.randint(0, self.num_zones)
                self.demand_queue[zone].append(dest_zone)
    
    def _match_and_dispatch(self):
        """Match vehicles to rides and calculate revenue for this step"""
        step_revenue = 0.0
        
        for zone in range(self.num_zones):
            # Find idle vehicles in zone
            idle_vehicles = np.where((self.vehicle_states[:, 0] == zone) & (self.vehicle_states[:, 1] == 1))[0]
            
            # Match with demand
            num_matches = min(len(idle_vehicles), len(self.demand_queue[zone]))
            
            for i in range(num_matches):
                vehicle_id = idle_vehicles[i]
                dest_zone = self.demand_queue[zone].pop(0)
                
                # Calculate revenue
                distance = euclidean(self.zone_positions[zone], self.zone_positions[dest_zone])
                base_fare = 2.5 + distance * 0.25  # Base + distance-based
                fare = base_fare * self.price_multipliers[zone]
                
                step_revenue += fare
                self.total_revenue += fare
                self.rides_completed += 1
                
                # Mark vehicle busy for this step only
                self.vehicle_states[vehicle_id, 1] = 0
                # After completing ride, vehicle instantly becomes idle at destination
                self.vehicle_states[vehicle_id, 0] = dest_zone
                self.vehicle_states[vehicle_id, 1] = 1

                
                # Note: ride completion not modeled over multiple steps for simplicity
        
        return step_revenue
    
    def _reposition_vehicles(self):
        """Reposition idle vehicles to areas with demand.
        Returns number of empty repositioning moves performed during this step.
        """
        step_empty = 0
        for zone in range(self.num_zones):
            idle_vehicles = np.where((self.vehicle_states[:, 0] == zone) & (self.vehicle_states[:, 1] == 1))[0]
            
            for vehicle_id in idle_vehicles:
                # 30% chance to reposition
                if np.random.random() < 0.3:
                    # Find zone with most demand
                    demand_per_zone = np.array([len(self.demand_queue[z]) for z in range(self.num_zones)])
                    new_zone = int(np.argmax(demand_per_zone))
                    
                    if new_zone != zone:
                        self.vehicle_states[vehicle_id, 0] = new_zone
                        step_empty += 1
        
        return step_empty
    
    def get_metrics(self):
        """Get cumulative performance metrics"""
        return {
            'total_revenue': self.total_revenue,
            'rides_completed': self.rides_completed,
            'empty_trips': self.empty_trips,
            'avg_revenue_per_ride': self.total_revenue / max(1, self.rides_completed)
        }