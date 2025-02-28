import numpy as np
import aerosandbox as asb
import matplotlib.pyplot as plt

def get_eqn(airfoil_name, no_deg=4):
    
    af = asb.Airfoil(airfoil_name)
    # af.draw()

    velocity = np.linspace(0.001, 25, 10)
    aero_analysis = asb.AeroBuildup(
        airplane = asb.Airplane(
            wings=[
                asb.Wing(
                    name="Main Wing",
                    xsecs=[
                        asb.WingXSec(
                            xyz_le=[0, 0, 0],
                            chord=1.5/5,
                            airfoil=af
                        ),
                        asb.WingXSec(
                            xyz_le=[0, 1, 0],
                            chord=1.5/5,
                            airfoil=af
                        )
                    ]
                )
            ]
        ),
        op_point = asb.OperatingPoint(
            velocity=velocity,
            alpha=5,
            beta=0,
            p=0,
            q=0,
            r=0,
        )
    )

    aero_analysis.airplane.draw()
    

    results = aero_analysis.run()

    forces_geometry = results["F_g"]
    forces_body = results["F_b"]
    forces_wind = results["F_w"]

    lift = results["L"]
    side_force = results["Y"]
    drag = results["D"]


    # print("Forces in geometry axes:", forces_geometry)
    print("Forces in body axes:", forces_body)
    # print("Forces in wind axes:", forces_wind)

    poly_fx = np.polyfit(velocity, forces_body[0], no_deg)
    poly_fy = np.polyfit(velocity, forces_body[1], no_deg)
    poly_fz = np.polyfit(velocity, forces_body[2], no_deg)

    # poly_fb = np.polyfit(velocity, forces_body, no_deg)

    fx_eq = np.poly1d(poly_fx)
    fy_eq = np.poly1d(poly_fy)
    fz_eq = np.poly1d(poly_fz)

    poly_lift = np.polyfit(velocity, lift, no_deg)
    poly_drag = np.polyfit(velocity, drag, no_deg)
    lift_eq = np.poly1d(poly_lift)
    drag_eq = np.poly1d(poly_drag)

    velocity2 = velocity
    velocity = np.linspace(0.001, 25, 10000)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Compute polynomial values
    poly_lift_computed = lift_eq(velocity)
    poly_drag_computed = drag_eq(velocity)
    
    # Plot both lift and drag on the same chart
    plt.plot(velocity, poly_lift_computed, 'b-', label="Lift Polynomial")
    plt.plot(velocity2, lift, 'bo', label="Lift Data")
    plt.plot(velocity, poly_drag_computed, 'r-', label="Drag Polynomial")
    plt.plot(velocity2, drag, 'ro', label="Drag Data")
    
    plt.legend()
    plt.xlabel("Velocity")
    plt.ylabel("Force")
    plt.title(("Lift and Drag vs Velocity", airfoil_name))
    plt.grid(True)
    plt.show()
    #plot the polynomials
    # plt.plot(velocity, forces_geometry[0], 'o')
    # plt.plot(velocity, fx_eq(velocity), '-')
    # # show original vals
    # plt.plot(velocity, forces_geometry[1], 'o')
    # plt.plot(velocity, fy_eq(velocity), '-')
    # # show original vals
    # plt.plot(velocity, forces_geometry[2], 'o')
    # plt.plot(velocity, fz_eq(velocity), '-')
    
    plt.show()
    

    print("Forces in body axes polynomials:", fx_eq, fy_eq, fz_eq)
    print("Lift polynomial:", lift_eq)
    print("Drag polynomial:", drag_eq)

    # print("Lift:", lift)
    # print("Side Force:", side_force)
    # print("Drag:", drag)

    # return poly_fb, poly_lift, poly_drag
    return fx_eq, fy_eq, fz_eq, lift_eq, drag_eq



get_eqn("ag35")