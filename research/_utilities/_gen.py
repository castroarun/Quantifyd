# Generator script
import pathlib
target = pathlib.Path(r"c:/Users/Castro/Documents/Projects/Covered_Calls/run_agent3_technical_optimization.py")
target.write_text(open(r"c:/Users/Castro/Documents/Projects/Covered_Calls/_script_template.txt").read())
print("Done")
