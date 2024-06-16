# Dedicated to cosmos
# NASODnsgaSO-NET

## Neural Architecture Search for Object Detectors using NSGA2 by Surrogates Optimization
## Genetic Integer Encoding

<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/0c6293b3-aeb7-43ef-8029-77e98053ced6" alt="image" width="1000">
<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/8f00c525-9b02-4461-b393-3d55200f452b" alt="image" width="1000">
<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/fddce117-cff6-4af5-9be6-00274717754d" alt="image" width="1000">
<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/fcff47e1-99f5-4bc2-ba12-16d6598d54ba" alt="image" width="1000">

## Novel Genomes
<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/04f40d88-5666-4fa6-8f09-10accfeb1953" alt="image" width="1000">

## Run Search 
```
# GENERATE INITIAL POPULATION
python3 popu_generator.py
```
```
# GENERATE ARCHIVE
# SINGLE GPU TRAINING
CUDA_VISIBLE_DEVICES=0 python3 archive.py
# MULTI GPU TRAINING
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 archive.py
```
```
# SINGLE GPU TRAINING
CUDA_VISIBLE_DEVICES=0 python3 nasod_gen_search.py
# MULTI GPU TRAINING
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 nasod_gen_search.py
```

## Genetic Pareto-front
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/f7359dde-c948-4378-bd9a-c383fefb840e" alt="image" width="1000">

## Dimension reduction of Genomes using AUTOENCODERS
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/9252a93a-3b90-4957-b4f9-63f9900b3741" alt="image" width="1000">

<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/bbb0169b-7f3d-42e4-bee7-29e287d9a8e3" alt="image" width="1000">

## Surrogate Models on Archive
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/0af960e3-c743-4c5e-8fdd-f668d400dc5b" alt="image" width="1000">

## Ancestors taken from different parts of Search Space
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/319f3ea1-d392-491b-97a9-d64b7a2f7408" alt="image" width="1000">

## Node Operation
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/21f7cdaa-ac61-4c71-9550-6915d28c1d3c" alt="image" width="1000">

## PASCAL-VOC 20 Class
<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/b57ed683-b944-4f8d-bf59-7f9a2ecd3532" alt="image" width="1000">

## MS-COCO 91 Class
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/de8f863c-2c2a-4300-89a2-a3636653e00f" alt="image" width="1000">

## Summary
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/4eb3d814-80ac-4ae1-8bf8-5d4b185d018c" alt="image" width="1000">

## All models are present in release
<img src="https://github.com/anuragGUPTA2235/NASOD-NET/assets/161227082/15300dc3-588b-447c-855e-6a8b9fd4710c" alt="image" width="1000">

## Archive
<img src="https://github.com/anuragGUPTA2235/NASODnsgaSO-NET/assets/161227082/c0cb35cd-8be6-4345-8938-b71b9164301b" alt="image" width="1000">
