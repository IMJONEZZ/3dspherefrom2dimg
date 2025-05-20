# 3dspherefrom2dimg
Process:
1. Come up with a hypothesis about how to estimate based on some cursory research
2. Implement my own code
3. Ask several LLMs (including my own) to check my work and make improvements
4. Test

Current Implementation:
Complete, runs, but I am unsatisfied with the results

Future iterations:
- radius + circumference estimation cast onto 3d (I think this will be better)


To run:
`pixi install`
`pixi run python fixed_sphere_code.py` (change example_sphere.png to be a given sphere)

To test:
`pixi run python -m unittest discover -s tests`