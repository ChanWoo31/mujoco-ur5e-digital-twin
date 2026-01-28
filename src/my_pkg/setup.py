from setuptools import find_packages, setup

package_name = 'my_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='han',
    maintainer_email='cwcw0301@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dynamixel_val = my_pkg.dynamixel_val:main',
            'ur5e_run = my_pkg.ur5e_run:main',
            'jacobian_ur5e_run_without_orientation = my_pkg.jacobian_ur5e_run_without_orientation:main',
            'jacobian_ur5e_run_with_orientation = my_pkg.jacobian_ur5e_run_with_orientation:main',
        ],
    },
)
