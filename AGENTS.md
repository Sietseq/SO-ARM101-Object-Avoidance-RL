# Project guidance

Read PROJECT_SPECS.md and README.md before making changes.
For ROS 2 environment, build, run, and inspection commands, consult
ROS2_COMMANDS.md and verify commands against the installed distribution and
the current workspace packages before documenting them as functional.

## Hiwonder hardware and LeRobot

- This project uses a Hiwonder SO-ARM101 with HX-30HM servos.
- Use Hiwonder's LeRobot fork for its motor drivers and hardware support:
  https://github.com/Hiwonder-official/hiwonder-SoArm-101
- The Python package is still named `lerobot`. Verify which checkout
  the active environment imports before changing dependencies.
- Do not replace the fork with upstream LeRobot without preserving
  and validating motor communication and calibration compatibility.
- The fork can lag upstream software features and fixes. Check the
  installed version before using commands or APIs from current docs.
- Future upgrades may involve porting Hiwonder hardware support to
  newer LeRobot. Keep robot communication separate from simulation
  and policy code to make that migration manageable.
- Compute and data requirements depend on the model and task;
  they are not inherently caused by using Hiwonder hardware.

## Privacy

- Keep personal information, absolute machine paths, credentials,
  and local editor settings out of committed files.
- Use placeholders or relative paths in shared documentation.
