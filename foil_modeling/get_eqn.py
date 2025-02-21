import aerosandbox as asb
import numpy as np
import matplotlib.pyplot as plt

af = asb.Airfoil("naca0012")
# af.draw()

velocity = np.linspace(5, 25, 5)
aero_analysis = asb.AeroBuildup(
    airplane = asb.Airplane(
        wings=[
            asb.Wing(
                name="Main Wing",
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=1.5,
                        airfoil=af
                    ),
                    asb.WingXSec(
                        xyz_le=[0, 5, 0],
                        chord=1.5,
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

results = aero_analysis.run()

forces_geometry = results["F_g"]
forces_body = results["F_b"]
forces_wind = results["F_w"]

print("Forces in geometry axes:", forces_geometry)
print("Forces in body axes:", forces_body)
print("Forces in wind axes:", forces_wind)



lift = results["L"]
side_force = results["Y"]
drag = results["D"]

print("Lift:", lift)
print("Side Force:", side_force)
print("Drag:", drag)


def get_eqn(wing_name, no_deg=4):
    
    af = asb.Airfoil(wing_name)
    # af.draw()

    velocity = np.linspace(5, 25, 5)
    aero_analysis = asb.AeroBuildup(
        airplane = asb.Airplane(
            wings=[
                asb.Wing(
                    name="Main Wing",
                    xsecs=[
                        asb.WingXSec(
                            xyz_le=[0, 0, 0],
                            chord=1.5,
                            airfoil=af
                        ),
                        asb.WingXSec(
                            xyz_le=[0, 5, 0],
                            chord=1.5,
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

    results = aero_analysis.run()

    forces_geometry = results["F_g"]
    forces_body = results["F_b"]
    forces_wind = results["F_w"]

    lift = results["L"]
    side_force = results["Y"]
    drag = results["D"]


    print("Forces in geometry axes:", forces_geometry)
    print("Forces in body axes:", forces_body)
    print("Forces in wind axes:", forces_wind)

    poly_fx = np.polyfit(velocity, forces_geometry[0], no_deg)
    poly_fy = np.polyfit(velocity, forces_geometry[1], no_deg)
    poly_fz = np.polyfit(velocity, forces_geometry[2], no_deg)

    fx_eq = np.poly1d(poly_fx)
    fy_eq = np.poly1d(poly_fy)
    fz_eq = np.poly1d(poly_fz)

    poly_lift = np.polyfit(velocity, lift, no_deg)
    poly_drag = np.polyfit(velocity, drag, no_deg)
    lift_eq = np.poly1d(poly_lift)
    drag_eq = np.poly1d(poly_drag)

    # velocity2 = velocity
    # velocity = np.linspace(5, 25, 10000)
    # poly_lift_computed = poly_lift[0]*velocity**4 + poly_lift[1]*velocity**3 + poly_lift[2]*velocity**2 + poly_lift[3]*velocity + poly_lift[4]
    # plt.plot(velocity, poly_lift_computed, 'c')
    # plt.plot(velocity2, lift, 'o')
    # plt.show()

    
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
    

    print("Forces in geometry axes polynomials:", fx_eq, fy_eq, fz_eq)
    print("Lift polynomial:", lift_eq)
    print("Drag polynomial:", drag_eq)

    # print("Lift:", lift)
    # print("Side Force:", side_force)
    # print("Drag:", drag)

    return fx_eq, fy_eq, fz_eq, lift_eq, drag_eq



get_eqn("naca0012")