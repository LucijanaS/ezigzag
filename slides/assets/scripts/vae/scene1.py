from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene1(VoiceoverScene):
    def construct(self):
        self.wait(2)

        # VAE architecture
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1.4, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1.4, 0], color=PURPLE
        )
        encoder_txt = Text("Encoder", color=WHITE).scale(0.6).move_to(encoder)

        mu = Rectangle(height=0.6, width=0.45, color=YELLOW)
        mu_txt = MathTex(r"\mu").scale(0.8).move_to(mu)
        sigma = Rectangle(height=0.6, width=0.45, color=YELLOW)
        sigma_txt = MathTex(r"\sigma").scale(0.8).move_to(sigma)

        mu = VGroup(mu, mu_txt)
        sigma = VGroup(sigma, sigma_txt)
        params = VGroup(mu, sigma).arrange(DOWN, buff=0.1).next_to(encoder, RIGHT, 0.25)

        distrib = Rectangle(height=1.2, width=1.8, color=RED).next_to(
            params, RIGHT, buff=0.25
        )
        distrib_txt = MathTex(r"\mathcal{N}(\mu, \sigma)").scale(0.8).move_to(distrib)
        distrib = VGroup(distrib, distrib_txt)

        sample = MathTex(r"\sim").scale(0.8).next_to(distrib, RIGHT, buff=0.2)

        z = Rectangle(height=1.2, width=0.45, color=BLUE).next_to(
            sample, RIGHT, buff=0.2
        )
        z_txt = MathTex(r"z").scale(0.8).move_to(z)
        z = VGroup(z, z_txt)

        decoder = Polygon(
            [-1, 0.6, 0], [1, 1.4, 0], [1, -1.4, 0], [-1, -0.6, 0], color=PURPLE
        )
        decoder.next_to(z, RIGHT, buff=0.2)
        decoder_txt = Text("Decoder", color=WHITE).scale(0.6).move_to(decoder)

        vae = VGroup(
            encoder, params, distrib, sample, z, decoder, encoder_txt, decoder_txt
        ).move_to(ORIGIN)

        vae_title = Text("Variational Autoencoder", color=WHITE).to_edge(UP)

        self.play(Write(vae_title))
        self.play(Create(vae), run_time=3)

        self.play(
            ShowPassingFlash(
                SurroundingRectangle(vae, buff=0.4, color=WHITE), time_width=0.4
            ),
            run_time=2,
        )

        self.wait(0.6)

        # Autoencoder architecture
        self.next_section(skip_animations=False)

        self.play(
            LaggedStart(
                FadeOut(distrib, distrib_txt, sample, params),
                AnimationGroup(
                    z.animate.move_to(ORIGIN),
                    VGroup(encoder, encoder_txt).animate.move_to(1.5 * LEFT),
                    VGroup(decoder, decoder_txt).animate.move_to(1.5 * RIGHT),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.play(FadeOut(vae_title, z, encoder, decoder, encoder_txt, decoder_txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1()
    scene.render()
