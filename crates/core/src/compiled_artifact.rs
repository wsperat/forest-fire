use super::*;

pub(crate) const COMPILED_ARTIFACT_MAGIC: [u8; 4] = *b"FFCA";
pub(crate) const COMPILED_ARTIFACT_VERSION: u16 = 1;
pub(crate) const COMPILED_ARTIFACT_BACKEND_CPU: u16 = 1;
pub(crate) const COMPILED_ARTIFACT_HEADER_LEN: usize = 8;

#[derive(Debug)]
pub enum CompiledArtifactError {
    ArtifactTooShort { actual: usize, minimum: usize },
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u16),
    UnsupportedBackend(u16),
    Encode(String),
    Decode(String),
    InvalidSemanticModel(IrError),
    InvalidRuntime(OptimizeError),
}

impl Display for CompiledArtifactError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CompiledArtifactError::ArtifactTooShort { actual, minimum } => write!(
                f,
                "Compiled artifact is too short: expected at least {} bytes, found {}.",
                minimum, actual
            ),
            CompiledArtifactError::InvalidMagic(magic) => {
                write!(f, "Compiled artifact has invalid magic bytes: {:?}.", magic)
            }
            CompiledArtifactError::UnsupportedVersion(version) => {
                write!(f, "Unsupported compiled artifact version: {}.", version)
            }
            CompiledArtifactError::UnsupportedBackend(backend) => {
                write!(f, "Unsupported compiled artifact backend: {}.", backend)
            }
            CompiledArtifactError::Encode(message) => {
                write!(f, "Failed to encode compiled artifact: {}.", message)
            }
            CompiledArtifactError::Decode(message) => {
                write!(f, "Failed to decode compiled artifact: {}.", message)
            }
            CompiledArtifactError::InvalidSemanticModel(err) => err.fmt(f),
            CompiledArtifactError::InvalidRuntime(err) => err.fmt(f),
        }
    }
}

impl Error for CompiledArtifactError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompiledArtifactPayload {
    pub(crate) semantic_ir: ModelPackageIr,
    pub(crate) runtime: OptimizedRuntime,
    #[serde(default)]
    pub(crate) feature_projection: Option<Vec<usize>>,
}

impl OptimizedModel {
    pub fn serialize_compiled(&self) -> Result<Vec<u8>, CompiledArtifactError> {
        let payload = CompiledArtifactPayload {
            semantic_ir: self.source_model.to_ir(),
            runtime: self.runtime.clone(),
            feature_projection: Some(self.feature_projection.clone()),
        };
        let mut payload_bytes = Vec::new();
        ciborium::into_writer(&payload, &mut payload_bytes)
            .map_err(|err| CompiledArtifactError::Encode(err.to_string()))?;
        let mut bytes = Vec::with_capacity(COMPILED_ARTIFACT_HEADER_LEN + payload_bytes.len());
        bytes.extend_from_slice(&COMPILED_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&COMPILED_ARTIFACT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&COMPILED_ARTIFACT_BACKEND_CPU.to_le_bytes());
        bytes.extend_from_slice(&payload_bytes);
        Ok(bytes)
    }

    pub fn deserialize_compiled(
        serialized: &[u8],
        physical_cores: Option<usize>,
    ) -> Result<Self, CompiledArtifactError> {
        if serialized.len() < COMPILED_ARTIFACT_HEADER_LEN {
            return Err(CompiledArtifactError::ArtifactTooShort {
                actual: serialized.len(),
                minimum: COMPILED_ARTIFACT_HEADER_LEN,
            });
        }

        let magic = [serialized[0], serialized[1], serialized[2], serialized[3]];
        if magic != COMPILED_ARTIFACT_MAGIC {
            return Err(CompiledArtifactError::InvalidMagic(magic));
        }

        let version = u16::from_le_bytes([serialized[4], serialized[5]]);
        if version != COMPILED_ARTIFACT_VERSION {
            return Err(CompiledArtifactError::UnsupportedVersion(version));
        }

        let backend = u16::from_le_bytes([serialized[6], serialized[7]]);
        if backend != COMPILED_ARTIFACT_BACKEND_CPU {
            return Err(CompiledArtifactError::UnsupportedBackend(backend));
        }

        let payload: CompiledArtifactPayload = ciborium::from_reader(std::io::Cursor::new(
            &serialized[COMPILED_ARTIFACT_HEADER_LEN..],
        ))
        .map_err(|err| CompiledArtifactError::Decode(err.to_string()))?;
        let source_model = ir::model_from_ir(payload.semantic_ir)
            .map_err(CompiledArtifactError::InvalidSemanticModel)?;
        let feature_projection = payload
            .feature_projection
            .unwrap_or_else(|| (0..source_model.num_features()).collect());
        let thread_count = resolve_inference_thread_count(physical_cores)
            .map_err(CompiledArtifactError::InvalidRuntime)?;
        let executor =
            InferenceExecutor::new(thread_count).map_err(CompiledArtifactError::InvalidRuntime)?;

        Ok(Self {
            source_model,
            runtime: payload.runtime,
            executor,
            feature_projection,
        })
    }
}
