import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Arrow
from scipy.integrate import odeint

class SHM:

    def __init__(self, mass, spring_constant, initial_displacement, initial_velocity=0):
        self.m = mass
        self.k = spring_constant
        self.x0 = initial_displacement
        self.v0 = initial_velocity
        self.omega = np.sqrt(self.k / self.m)
        self.A = np.sqrt(self.x0**2 + (self.v0/self.omega)**2)
        self.phase = np.arctan2(-self.v0 / self.omega, self.x0)


    @staticmethod
    def coupled_oscillators(m1, m2, k1, k2, k3, x1_0, x2_0, v1_0=0, v2_0=0, t_max=10, num_points=1000):
        def derivs(state, t):
            x1, v1, x2, v2 = state
            dxdt1 = v1
            dxdt2 = v2
            dvdt1 = (-k1*x1 - k2*(x1-x2)) / m1
            dvdt2 = (-k3*x2 - k2*(x2-x1)) / m2
            return [dxdt1, dvdt1, dxdt2, dvdt2]

        t = np.linspace(0, t_max, num_points)
        initial_state = [x1_0, v1_0, x2_0, v2_0]
        solution = odeint(derivs, initial_state, t)
        x1, x2 = solution[:, 0], solution[:, 2]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, x1, label='Mass 1')
        ax.plot(t, x2, label='Mass 2')
        ax.set_xlabel('Time')
        ax.set_ylabel('Displacement')
        ax.set_title('Coupled Oscillators')
        ax.legend()
        plt.show()

        return t, x1, x2

    @staticmethod
    def animate_coupled_oscillators(t, x1, x2):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True)

        mass1, = ax.plot([], [], 'ro', markersize=20)
        mass2, = ax.plot([], [], 'bo', markersize=20)
        spring1, = ax.plot([], [], 'k-')
        spring2, = ax.plot([], [], 'k-')
        spring3, = ax.plot([], [], 'k-')

        def init():
            mass1.set_data([], [])
            mass2.set_data([], [])
            spring1.set_data([], [])
            spring2.set_data([], [])
            spring3.set_data([], [])
            return mass1, mass2, spring1, spring2, spring3

        def animate(i):
            mass1.set_data([x1[i]], [0])  # Note the change here
            mass2.set_data([x2[i]], [0])  # And here
            spring1.set_data([-2, x1[i]], [0, 0])
            spring2.set_data([x1[i], x2[i]], [0, 0])
            spring3.set_data([x2[i], 2], [0, 0])
            return mass1, mass2, spring1, spring2, spring3

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=20, blit=True)
        plt.show()

    @staticmethod
    def tension_in_string(m1, m2, L, theta_max, num_points=1000):
        g = 9.8
        theta = np.linspace(-theta_max, theta_max, num_points)
        
        T1 = m1 * g * (3 - 2 * np.cos(theta))
        T2 = m2 * g * (3 - 2 * np.cos(theta))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.degrees(theta), T1, label='Tension in string 1')
        ax.plot(np.degrees(theta), T2, label='Tension in string 2')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Tension (N)')
        ax.set_title('Tension in Strings vs Angle')
        ax.legend()
        plt.grid(True)
        plt.show()

        return theta, T1, T2

    @staticmethod
    def visualize_tension_system(m1, m2, L, theta):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-L*1.2, L*1.2)
        ax.set_ylim(-L*1.2, L*0.2)
        ax.set_aspect('equal')
        ax.plot(0, 0, 'ko', markersize=10)
        x1 = L * np.sin(theta)
        y1 = -L * np.cos(theta)
        x2 = 2 * x1
        y2 = 2 * y1

        ax.plot([0, x1, x2], [0, y1, y2], 'k-')

        
        mass1 = Circle((x1, y1), 0.05*L, fill=True, color='r')
        mass2 = Circle((x2, y2), 0.05*L, fill=True, color='b')
        ax.add_artist(mass1)
        ax.add_artist(mass2)

        
        ax.text(x1, y1-0.15*L, f'm1 = {m1} kg', ha='center')
        ax.text(x2, y2-0.15*L, f'm2 = {m2} kg', ha='center')
        ax.text(0, 0.1*L, f'θ = {np.degrees(theta):.1f}°', ha='center')

       
        tension_scale = 0.2 * L
        t1_x = tension_scale * np.sin(theta)
        t1_y = tension_scale * np.cos(theta)
        t2_x = tension_scale * np.sin(theta)
        t2_y = tension_scale * np.cos(theta)

        ax.arrow(x1, y1, t1_x, t1_y, head_width=0.05*L, head_length=0.1*L, fc='g', ec='g')
        ax.arrow(x2, y2, t2_x, t2_y, head_width=0.05*L, head_length=0.1*L, fc='g', ec='g')

        ax.text(x1+1.2*t1_x, y1+1.2*t1_y, 'T1', color='g')
        ax.text(x2+1.2*t2_x, y2+1.2*t2_y, 'T2', color='g')

        plt.title('Two Masses Connected by Strings')
        ax.set_axis_off()
        plt.show()

    def displacement(self, t):
        return self.A * np.cos(self.omega * t + self.phase)

    def velocity(self, t):
        return -self.A * self.omega * np.sin(self.omega * t + self.phase)

    def acceleration(self, t):
        return -self.A * self.omega**2 * np.cos(self.omega * t + self.phase)

    def energy_kinetic(self, t):
        return 0.5 * self.m * self.velocity(t)**2

    def energy_potential(self, t):
        return 0.5 * self.k * self.displacement(t)**2

    def energy_total(self, t):
        return self.energy_kinetic(t) + self.energy_potential(t)

    def period(self):
        return 2 * np.pi / self.omega

    def frequency(self):
        return 1 / self.period()

    def visualize_motion(self, t_max, num_points=1000):
        t = np.linspace(0, t_max, num_points)
        x = self.displacement(t)
        v = self.velocity(t)
        a = self.acceleration(t)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        ax1.plot(t, x)
        ax1.set_title('Displacement vs Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Displacement (m)')
        ax1.grid(True)

        ax2.plot(t, v)
        ax2.set_title('Velocity vs Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True)

        ax3.plot(t, a)
        ax3.set_title('Acceleration vs Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

    def visualize_energy(self, t_max, num_points=1000):
        t = np.linspace(0, t_max, num_points)
        ke = [self.energy_kinetic(ti) for ti in t]
        pe = [self.energy_potential(ti) for ti in t]
        te = [self.energy_total(ti) for ti in t]

        plt.figure(figsize=(10, 6))
        plt.plot(t, ke, label='Kinetic Energy')
        plt.plot(t, pe, label='Potential Energy')
        plt.plot(t, te, label='Total Energy')
        plt.title('Energy vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_phase_space(self, t_max, num_points=1000):
        t = np.linspace(0, t_max, num_points)
        x = self.displacement(t)
        v = self.velocity(t)

        plt.figure(figsize=(8, 8))
        plt.plot(x, v)
        plt.title('Phase Space Diagram')
        plt.xlabel('Displacement (m)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def visualize_free_body_diagram(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw mass
        mass = Circle((0, 0), 0.2, fill=False)
        ax.add_artist(mass)
        
        # Draw spring
        spring_x = np.linspace(-1, 0, 20)
        spring_y = 0.05 * np.sin(20 * spring_x)
        ax.plot(spring_x, spring_y, 'k-')
        
        # Draw forces
        spring_force = Arrow(0, 0, -0.5, 0, width=0.05, color='r')
        ax.add_artist(spring_force)
        
        # Labels
        ax.text(0, 0.3, 'm', ha='center', va='center')
        ax.text(-0.25, 0.1, 'F = -kx', color='r', ha='center', va='center')
        
        ax.set_xlim(-1.2, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Free Body Diagram - SHM')
        plt.show()

class NLM:
    @staticmethod
    def force_calculator(mass, acceleration):
        return mass * acceleration

    @staticmethod
    def weight(mass, g=9.8):
        return mass * g

    @staticmethod
    def friction_force(normal_force, mu):
        return mu * normal_force

    @staticmethod
    def centripetal_force(mass, velocity, radius):
        return mass * velocity**2 / radius

    @staticmethod
    def solve_inclined_plane(mass, angle, mu=0):
        g = 9.8
        theta = np.radians(angle)
        
        N = mass * g * np.cos(theta)
        F_friction = NLM.friction_force(N, mu)
        F_gravity_parallel = mass * g * np.sin(theta)
        
        F_net = F_gravity_parallel - F_friction
        acceleration = F_net / mass
        
        return {
            'Normal Force': N,
            'Friction Force': F_friction,
            'Net Force': F_net,
            'Acceleration': acceleration
        }

    @staticmethod
    def visualize_inclined_plane(mass, angle, mu=0):
        results = NLM.solve_inclined_plane(mass, angle, mu)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw inclined plane
        plane_length = 5
        x = np.cos(np.radians(angle)) * plane_length
        y = np.sin(np.radians(angle)) * plane_length
        ax.plot([0, x], [0, y], 'k-', linewidth=2)
        ax.plot([0, plane_length], [0, 0], 'k-', linewidth=2)
        
        # Draw mass
        mass_x = x / 2
        mass_y = y / 2
        mass_circle = Circle((mass_x, mass_y), 0.2, fill=False)
        ax.add_artist(mass_circle)
        
        # Draw forces
        scale = 0.5 / mass / 9.8  # Scale factor for force arrows
        
        # Weight
        weight = Arrow(mass_x, mass_y, 0, -scale * NLM.weight(mass), width=0.1, color='blue')
        ax.add_artist(weight)
        
        # Normal force
        normal = Arrow(mass_x, mass_y, -scale * results['Normal Force'] * np.sin(np.radians(angle)),
                       scale * results['Normal Force'] * np.cos(np.radians(angle)), width=0.1, color='green')
        ax.add_artist(normal)
        
        # Friction force
        friction = Arrow(mass_x, mass_y, -scale * results['Friction Force'] * np.cos(np.radians(angle)),
                         -scale * results['Friction Force'] * np.sin(np.radians(angle)), width=0.1, color='red')
        ax.add_artist(friction)
        
        # Labels
        ax.text(mass_x, mass_y - 0.5, f'm = {mass} kg', ha='center', va='center')
        ax.text(x/2, -0.5, f'θ = {angle}°', ha='center', va='center')
        ax.text(mass_x + 0.5, mass_y, 'N', color='green', ha='left', va='center')
        ax.text(mass_x - 0.5, mass_y - 0.3, 'W', color='blue', ha='right', va='center')
        ax.text(mass_x - 0.7, mass_y, 'f', color='red', ha='right', va='center')
        
        ax.set_xlim(-1, plane_length + 1)
        ax.set_ylim(-1, y + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Free Body Diagram - Inclined Plane')
        plt.show()

def main():
    while True:
        print("\nPhysics Problem Solver")
        print("1. Simple Harmonic Motion (SHM)")
        print("2. Newton's Laws of Motion (NLM)")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            shm_menu()
        elif choice == '2':
            nlm_menu()
        elif choice == '3':
            print("Using homework(physics) assistant")
            break
        else:
            print("Invalid choice. Please try again.")

def shm_menu():
    print("\nSimple Harmonic Motion (SHM)")
    
    while True:
        print("\nSHM Options:")
        print("1. Single oscillator")
        print("2. Coupled oscillators")
        print("3. Tension in string system")
        print("4. Return to main menu")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            single_oscillator_menu()
        elif choice == '2':
            coupled_oscillators_menu()
        elif choice == '3':
            tension_in_string_menu()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

def single_oscillator_menu():
    mass = float(input("Enter mass (kg): "))
    spring_constant = float(input("Enter spring constant (N/m): "))
    initial_displacement = float(input("Enter initial displacement (m): "))
    initial_velocity = float(input("Enter initial velocity (m/s): "))
    
    shm = SHM(mass, spring_constant, initial_displacement, initial_velocity)
    
    while True:
        print("\nSingle Oscillator Options:")
        print("1. Calculate period and frequency")
        print("2. Visualize motion")
        print("3. Visualize energy")
        print("4. Visualize phase space")
        print("5. Visualize free body diagram")
        print("6. Return to SHM menu")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            print(f"Period: {shm.period():.4f} s")
            print(f"Frequency: {shm.frequency():.4f} Hz")
        elif choice == '2':
            shm.visualize_motion(2 * shm.period())
        elif choice == '3':
            shm.visualize_energy(2 * shm.period())
        elif choice == '4':
            shm.visualize_phase_space(2 * shm.period())
        elif choice == '5':
            shm.visualize_free_body_diagram()
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")

def coupled_oscillators_menu():
    m1 = float(input("Enter mass 1 (kg): "))
    m2 = float(input("Enter mass 2 (kg): "))
    k1 = float(input("Enter spring constant 1 (N/m): "))
    k2 = float(input("Enter spring constant 2 (N/m): "))
    k3 = float(input("Enter spring constant 3 (N/m): "))
    x1_0 = float(input("Enter initial displacement of mass 1 (m): "))
    x2_0 = float(input("Enter initial displacement of mass 2 (m): "))

    t, x1, x2 = SHM.coupled_oscillators(m1, m2, k1, k2, k3, x1_0, x2_0)
    SHM.animate_coupled_oscillators(t, x1, x2)

def tension_in_string_menu():
    m1 = float(input("Enter mass 1 (kg): "))
    m2 = float(input("Enter mass 2 (kg): "))
    L = float(input("Enter string length (m): "))
    theta_max = float(input("Enter maximum angle (degrees): "))

    theta, T1, T2 = SHM.tension_in_string(m1, m2, L, np.radians(theta_max))
    
    while True:
        print("\nTension in String System Options:")
        print("1. Visualize tension graph")
        print("2. Visualize system at specific angle")
        print("3. Return to SHM menu")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            SHM.tension_in_string(m1, m2, L, np.radians(theta_max))
        elif choice == '2':
            angle = float(input("Enter angle for visualization (degrees): "))
            SHM.visualize_tension_system(m1, m2, L, np.radians(angle))
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

def nlm_menu():
    print("\nNewton's Laws of Motion (NLM)")
    
    while True:
        print("\nNLM Options:")
        print("1. Force calculator")
        print("2. Weight calculator")
        print("3. Friction force calculator")
        print("4. Centripetal force calculator")
        print("5. Inclined plane problem")
        print("6. Return to main menu")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            mass = float(input("Enter mass (kg): "))
            acceleration = float(input("Enter acceleration (m/s²): "))
            force = NLM.force_calculator(mass, acceleration)
            print(f"Force: {force:.2f} N")
        elif choice == '2':
            mass = float(input("Enter mass (kg): "))
            weight = NLM.weight(mass)
            print(f"Weight: {weight:.2f} N")
        elif choice == '3':
            normal_force = float(input("Enter normal force (N): "))
            mu = float(input("Enter coefficient of friction: "))
            friction = NLM.friction_force(normal_force, mu)
            print(f"Friction force: {friction:.2f} N")
        elif choice == '4':
            mass = float(input("Enter mass (kg): "))
            velocity = float(input("Enter velocity (m/s): "))
            radius = float(input("Enter radius (m): "))
            centripetal_force = NLM.centripetal_force(mass, velocity, radius)
            print(f"Centripetal force: {centripetal_force:.2f} N")
        elif choice == '5':
            mass = float(input("Enter mass (kg): "))
            angle = float(input("Enter angle of incline (degrees): "))
            mu = float(input("Enter coefficient of friction: "))
            results = NLM.solve_inclined_plane(mass, angle, mu)
            print("\nInclined Plane Results:")
            for key, value in results.items():
                print(f"{key}: {value:.2f}")
            NLM.visualize_inclined_plane(mass, angle, mu)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()