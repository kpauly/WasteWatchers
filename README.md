# WasteWatchers

This script takes a folder of drone-based, 5mm GSD RGB orthomosaic geotiffs as input and calls the RS-LLaVA VLM to classify each 224x224 px tile as either containing litter or not, in the framework of the Flemish amai! project [Waste Watchers](https://www.river-cleanup.org/nl/waste-watchers). The script then saves the outline of each positive tile (containing litter) as a polygon feature to a shapefile with the same filename as the corresponding orthomosaic. Each geotiff is processed on a separate available thread to speed up the process. As a prerequisite, [RS-LLaVA](https://github.com/BigData-KSU/RS-LLaVA) needs to be [installed](https://github.com/BigData-KSU/RS-LLaVA?tab=readme-ov-file#install).

## Usage
Change the folder path in line 126 to the desired location and the number of available threads in line 128 to a sensible number. Then run the script.
