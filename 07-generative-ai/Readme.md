### Generative AI

It's a branch of artificial intelligence that learns patterns from existing **data** to **generate new content**, including text, images, audio, video, code, and other media.

> Simply: `Generative AI = AI that creates.`

**Enterprise Use Cases**

| Industry      | Example                          |
| ------------- | -------------------------------- |
| Healthcare    | Clinical note drafting           |
| Banking       | Customer support assistants      |
| Education     | Personalized tutoring            |
| Software      | Code generation and review       |
| Marketing     | Ad copy and campaign ideas       |
| Legal         | Drafting contracts and summaries |
| Manufacturing | Technical documentation          |

**Traditional AI vs Generative AI**

| Traditional AI | Generative AI |
| -------------- | ------------- |
| Predicts       | Creates       |
| Classifies     | Generates     |
| Detects        | Produces      |
| Recognizes     | Synthesizes   |

### Types of Generative AI

| Traditional AI | Generative AI |
| -------------- | ------------- |
| Predicts       | Creates       |
| Classifies     | Generates     |
| Detects        | Produces      |
| Recognizes     | Synthesizes   |

#### Popular Generative Models

| Content Type | Common Model Types                 |
| ------------ | ---------------------------------- |
| Text         | LLMs (GPT, Llama, etc.)            |
| Images       | Diffusion Models                   |
| Audio        | Speech and audio generation models |
| Video        | Video generation models            |
| Code         | Code-specialized LLMs              |

#### Where Do LLMs Fit?

```bash
Artificial Intelligence
        ↓
Machine Learning
        ↓
Deep Learning
        ↓
Generative AI
        ├── Text (LLMs)
        ├── Images
        ├── Audio
        ├── Video
        └── Code
```

### Foundation Models

Foundation model is a large, pretrained AI model that learns general patterns from massive datasets and can be adapted for many different downstream tasks.

> Simply: `Foundation Model = A general-purpose AI model that serves as the base for many applications.`

| Foundation Model            | Fine-Tuned Model                    |
| --------------------------- | ----------------------------------- |
| General-purpose             | Specialized                         |
| Broad knowledge             | Domain-specific                     |
| Trained on massive datasets | Adapted on smaller datasets         |
| Reusable for many tasks     | Optimized for a particular use case |

### Diffusion Models

Diffusion model is a generative model that learns to generate new data by reversing a gradual process of adding noise.

> Simply: `Diffusion Model = Learn to remove noise until a new image appears.`

```bash
Original Image
      ↓
Add Small Noise
      ↓
Add More Noise
      ↓
Add More Noise
      ↓
Pure Random Noise
```

**Why Diffusion Models Became Popular** Earlier image-generation methods often struggled with stability or image quality. Diffusion models improved:

- High image quality
- Better prompt alignment
- Stable training
- Rich detail
- Greater diversity

> This is why they became the dominant approach for text-to-image generation.

### GANs vs VAEs vs Diffusion

#### Generative Adversarial Networks (GANs)

Generative adversarial (hostile, negative, antagonistic, adverse, contentious, adversary, conflicting, opposed) network contains two neural networks that compete with each other.

Generator: Creates fake images.

```bash
Random Noise
↓
Generator
↓
Fake Image
```

#### Variational Autoencoders (VAEs)

Variational autoencoder learns a compressed representation (latent space) of data and then reconstructs it.

Think of it as:

```bash
Image
↓
Compress
↓
Latent Space
↓
Decode
↓
Reconstructed Image
```

| Feature            | VAE             | GAN                           | Diffusion     |
| ------------------ | --------------- | ----------------------------- | ------------- |
| Main Idea          | Encode & Decode | Generator vs Discriminator    | Reverse Noise |
| Image Quality      | Moderate        | High                          | Very High     |
| Training Stability | High            | Low                           | High          |
| Diversity          | Good            | Can suffer from mode collapse | Excellent     |
| Inference Speed    | Fast            | Fast                          | Slower        |
| Prompt Control     | Limited         | Limited                       | Excellent     |
| Current Popularity | Moderate        | Moderate                      | Very High     |

### Image Generation

Image generation is the process of creating entirely new images from prompts or other inputs using AI models.

> Simply: `Image Generation = Prompt → AI → New Image`

Relation to LLMs

| LLM                           | Image Generator                                   |
| ----------------------------- | ------------------------------------------------- |
| Generates tokens              | Generates pixels (through latent representations) |
| Uses autoregressive decoding  | Uses iterative denoising                          |
| Output is text                | Output is an image                                |
| Transformer is the core model | Diffusion model is the core model                 |

### Text Generation

Text generation is the process of creating new natural language from a prompt using a language model.

> Simply: `Text Generation = Prompt → LLM → New Text`

### Code Generation

Code Generation is the use of Generative AI to create, explain, modify, optimize, or debug source code from natural language instructions.

> Simply: `Code Generation = Prompt → AI → Source Code`

### Audio Generation

Audio Generation is the use of Generative AI to create or transform audio, including: Speech, Music, Sound effects, Environmental sounds.

> Simply: `Audio Generation = Prompt or Audio → AI → New Audio`

### Video Generation

Video Generation is the use of Generative AI to create or modify videos from text, images, or existing videos.

> Simply: `Video Generation = Prompt → AI → Video`

