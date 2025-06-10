## AOI split logic

Instead of using geometric difference (which creates holes), it creates up to 4 rectangular boxes around the subtract geometry:


┌─────────────────────────────────┐
│           TOP BOX               │
├─────┬───────────────────┬───────┤
│LEFT │   SUBTRACT_GEOM   │ RIGHT │
│ BOX │                   │  BOX  │
├─────┴───────────────────┴───────┤
│          BOTTOM BOX             │
└─────────────────────────────────┐

The logic for this is defined in _split_geometry_into_boxes