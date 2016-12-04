# hypernsga

[![Build Status](https://travis-ci.org/mneumann/hypernsga.svg?branch=master)](https://travis-ci.org/mneumann/hypernsga)

Evolutionary Hyper-cube based neural net optimization using NSGA2 (Non Dominated Sorting Genetic Algorithm 2)

## Screenshot

![HyperNSGA screenshot](/doc/screenshot.png?raw=true "hypernsga")

Uses [imgui] branch.

## Installation

Requires Rust >= 1.9.0

## Running

```
git checkout imgui
cd hyperneat-gui
cargo run --release -- ../nets/skorpion.gml
```

[imgui]: https://github.com/mneumann/hypernsga/tree/imgui
