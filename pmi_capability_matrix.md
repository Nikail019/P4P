| PMI type | Parsed | Value | STEP link | Mapped | Displayed | Current scope | Notes |
|---|---|---|---|---|---|---|---|
| datums | partial | partial | partial | partial | partial | partial | DATUM/DATUM_FEATURE/DATUM_TARGET entities are parsed; label and face mapping can work, but formal datum feature symbol semantics are incomplete. |
| datum labels / datum features | partial | partial | partial | partial | partial | partial | Datum label extraction is simple string-based; display is approximate flag/leader. |
| linear distance dimensions | partial | yes | partial | partial | partial | partial | DIMENSIONAL_CHARACTERISTIC_REPRESENTATION is parsed. Two-plane display can be reconstructed along plane normal; cylinder/axis cases remain approximate. |
| diameter dimensions | partial | partial | partial | partial | partial | partial | May be parsed as a generic dimension if represented through DIMENSIONAL_*; diameter semantics are not explicitly classified yet. |
| radius dimensions | partial | partial | partial | partial | partial | partial | May be parsed as a generic dimension; radius-specific semantics are not fully separated from generic dimensions. |
| hole tolerances / hole size tolerances | partial | partial | partial | partial | partial | partial | Hole size may be represented as dimensions/modifiers; fit semantics need more parsing. |
| ISO IT grades | partial | partial | no | no | no | partial | Current code only searches modifiers for IT text; no robust ISO fit/grade parser. |
| general dimensional tolerances | partial | partial | partial | partial | partial | partial | PLUS_MINUS_TOLERANCE and modifiers are retained raw, not normalized into upper/lower tolerance fields. |
| geometric tolerances | partial | partial | partial | partial | partial | partial | Known tolerance entity names are recognized, but display is generic label/leader, not full FCF. |
| flatness | partial | partial | partial | partial | partial | partial | Recognized when entity is FLATNESS_TOLERANCE; not standards-complete. |
| straightness | partial | partial | partial | partial | partial | partial | Recognized when entity is STRAIGHTNESS_TOLERANCE; not standards-complete. |
| circularity | partial | partial | partial | partial | partial | partial | Code currently recognizes ROUNDNESS_TOLERANCE rather than explicitly naming circularity. |
| cylindricity | partial | partial | partial | partial | partial | partial | Recognized when entity is CYLINDRICITY_TOLERANCE; not standards-complete. |
| parallelism | partial | partial | partial | partial | partial | partial | Entity name recognized; datum reference handling is lightweight. |
| perpendicularity | partial | partial | partial | partial | partial | partial | Entity name recognized; datum reference handling is lightweight. |
| angularity | partial | partial | partial | partial | partial | partial | Entity name recognized; datum reference handling is lightweight. |
| position | partial | partial | partial | partial | partial | partial | POSITION_TOLERANCE recognized; feature control frame semantics are not normalized. |
| concentricity/coaxiality | no | no | no | no | no | no | Not explicitly in the tolerance keyword list. |
| symmetry | no | no | no | no | no | no | Not explicitly in the tolerance keyword list. |
| profile of a line | partial | partial | partial | partial | partial | partial | PROFILE_OF_A_LINE_TOLERANCE recognized generically. |
| profile of a surface | partial | partial | partial | partial | partial | partial | PROFILE_OF_A_SURFACE_TOLERANCE recognized generically. |
| runout / total runout | partial | partial | partial | partial | partial | partial | CIRCULAR_RUNOUT_TOLERANCE and TOTAL_RUNOUT_TOLERANCE recognized generically. |
| surface finish | no | no | no | no | no | no | Surface texture/finish entities are not parsed. |
