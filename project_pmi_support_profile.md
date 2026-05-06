# Project PMI Support Profile

## Level 1 - Supported for Current Project QA

- **datum labels**: label can be parsed, linked, mapped, and audited; display is approximate.
- **datum features**: datum feature face mapping can be audited when AP242 references resolve.
- **linear distance dimensions**: DIMENSIONAL_LOCATION records with mapped references are classified conservatively; two-plane displays can be QA-checked.

## Level 2 - Partially Supported

- **diameter dimensions**: detected by diameter text/symbol or cylinder diameter match; AP242 diameter semantics are not fully normalized.
- **radius dimensions**: detected by radius text or cylinder radius match; not fully normalized.
- **hole size dimensions**: detected when nominal value matches mapped cylinder diameter; hole fit semantics are partial.
- **general dimensional tolerances**: raw modifiers are retained; upper/lower bounds are best-effort only.
- **upper/lower tolerance bounds**: simple numeric extraction only; no full tolerance model.
- **ISO IT grades / IT6-IT10**: IT/fit text can be detected if present; ISO 286 meaning is not interpreted.
- **generic geometric tolerances**: detected and mapped, but not considered normalized enough for Level 1.
- **flatness**: entity name classified; FCF semantics incomplete.
- **straightness**: entity name classified; FCF semantics incomplete.
- **circularity**: ROUNDNESS_TOLERANCE classified as circularity; FCF semantics incomplete.
- **cylindricity**: entity name classified; FCF semantics incomplete.
- **parallelism**: entity name classified; datum frame reconstruction is partial.
- **perpendicularity**: entity name classified; datum frame reconstruction is partial.
- **angularity**: entity name classified; datum frame reconstruction is partial.
- **position**: entity name classified; target feature/tolerance zone semantics are partial.
- **profile**: profile entity names are grouped; details incomplete.
- **runout**: runout entity names are grouped; datum/axis semantics incomplete.

## Level 3 - Out of Current Scope

- **surface finish**: surface texture entities are not parsed.
- **full feature-control-frame rendering**: viewer is QA-oriented, not standards-complete MBD reconstruction.
- **full ISO 286 fit interpretation**: fit text may be detected, but fit meaning is not interpreted.
