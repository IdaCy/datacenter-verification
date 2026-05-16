# Observables

This directory is an independent observables-first rebuild for the datacenter
verification project.

The public taxonomy contains only OXX observable families and their feature
values. These are raw or source-emitted infrastructure values about capability,
activity, timing, physical state, resource use, and data movement inside a
monitored boundary.

The feature set is value-level: broad document wrappers, join identifiers,
source record identifiers, exact asset IDs, administrative IDs, ticket IDs, and
source references are not observable feature values. They may be implementation
data outside this taxonomy if needed to join records to a monitored site or
resource.

## Core Boundary

Observable feature values may show:

- installed and usable accelerator capacity;
- accelerator partitioning, clocks, caps, health, and state;
- resource allocation, reservation, provisioning, runtime, quota, and usage
  intervals;
- accelerator, fabric, network, storage, power, cooling, and environmental
  activity;
- maintenance, inventory, topology-change, physical-access, and storage-operation
  intervals/events that affect infrastructure state;
- site-specific utility, electrical, and asset timing values.

## Files

- `observables.yaml`: public OXX observable families and feature values.
- `sources.yaml`: source registry for measurement-surface existence and caveats.
- rules/ has own README