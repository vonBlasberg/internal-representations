# Original generation inputs

This directory documents the completed comparison. `request.json` records the submitted settings; `transport.json` inventories the source payload; `source_payload.anonymized.zip` contains the retained execution source and its dependencies. Paths, sizes and checksum references describe the anonymized derivative, as explained in [ANONYMIZATION.md](../../ANONYMIZATION.md). The private research archive retains the original submission and audits.

The archive is an execution record, not a portable launcher. Its one-use requests refer to the original environment, checkpoint location and provider setup. CPU replay uses the saved responses and does not import this GPU source. For a future generation run, [GPU_SETUP.md](../../GPU_SETUP.md) describes the resources and qualification needed. The supported runnable commands for this release are in the root [README.md](../../README.md).
