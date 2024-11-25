from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


def ddpm_scheme(y, x, t):
    t = max(0, min(t, 1))
    interp = (1-t) * x + t * y
    return interp.astype(np.uint8)


class Scene1(VoiceoverScene):
    def construct(self):
        self.wait(2)

        ddpm_title = Text("Diffusion", color=WHITE)

        img_x0 = (
            ImageMobject("../images/gas_tng50-1.519.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(1)
            .move_to(ORIGIN + 2.5 * LEFT)
        )
        img_x0_rect = SurroundingRectangle(img_x0, color=WHITE, buff=0.1)

        rgb_rnd = (np.random.rand(*img_x0.pixel_array.shape) * 255).astype(np.uint8)
        rgb_rnd[:, :, 3] = 255

        img_x1 = (
            ImageMobject(ddpm_scheme(img_x0.pixel_array, rgb_rnd, 0.5))
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(1)
            .move_to(ORIGIN + 2.5 * RIGHT)
        )
        img_x2 = (
            ImageMobject(ddpm_scheme(img_x0.pixel_array, rgb_rnd, 0.3))
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(1)
            .move_to(ORIGIN + 2.5 * RIGHT)
        )
        img_x3 = (
            ImageMobject(ddpm_scheme(img_x0.pixel_array, rgb_rnd, 0.1))
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(1)
            .move_to(ORIGIN + 2.5 * RIGHT)
        )
        img_xT = (
            ImageMobject(ddpm_scheme(img_x0.pixel_array, rgb_rnd, 0.0))
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(1)
            .move_to(ORIGIN + 2.5 * RIGHT)
        )

        txt_x0 = MathTex(r"x_0").move_to(ORIGIN + 2.5 * RIGHT)
        txt_x0_rect = img_x0_rect.copy().move_to(ORIGIN + 2.5 * RIGHT)

        arrow_x1 = Arrow(img_x0.get_right(), img_x1.get_left(), color=WHITE)
        def_eq = MathTex(r"\equiv").move_to(ORIGIN)
        
        self.next_section(skip_animations=False)
        self.play(Write(ddpm_title))
        self.wait(2)
        self.play(FadeOut(ddpm_title))
        self.wait(1)

        # Image = x0
        self.next_section(skip_animations=False)
        self.play(
            Create(img_x0_rect),
            FadeIn(img_x0),
            Create(txt_x0_rect),
            FadeIn(txt_x0)
        )
        self.play(GrowFromEdge(def_eq, LEFT))
        self.wait(3)

        # x0 -> x1
        self.play(FadeOut(img_x0_rect, def_eq, txt_x0_rect, txt_x0))
        self.play(
            FadeIn(img_x1),
            GrowArrow(arrow_x1)
        )
        self.wait(2)

        # x1 -> x2
        g_x0_x1 = Group(img_x0, arrow_x1, img_x1)
        self.play(
            LaggedStart(
                AnimationGroup(
                    g_x0_x1.animate.move_to(2.5/0.8 * LEFT).scale(0.8),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        arrow_x2 = arrow_x1.copy().shift(RIGHT * 4.05)
        img_x2 = img_x2.scale(0.8).move_to(RIGHT * 3)
        self.play(
            FadeIn(img_x2),
            GrowArrow(arrow_x2),
        )
        self.wait(2)

        # x2 -> x3
        g_x0_x2 = Group(img_x0, arrow_x1, img_x1, arrow_x2, img_x2)
        self.play(
            LaggedStart(
                AnimationGroup(
                    g_x0_x2.animate.move_to(2.5 * LEFT).scale(0.7),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        arrow_x3 = arrow_x2.copy().shift(RIGHT * 2.94)
        img_x3 = img_x3.scale(0.8*0.7).move_to(RIGHT * 3.25)
        self.play(
            FadeIn(img_x3),
            GrowArrow(arrow_x3),
        )
        self.wait(2)
        self.play(
            FadeOut(img_x0, arrow_x1, img_x1, arrow_x2, img_x2, arrow_x3, img_x3)
        )

        # Formulas
        self.next_section(skip_animations=False)
        forward_ddpm_formula = MathTex(
            r"q(x_{1:T}) \equiv \prod_{t=1}^T q(x_t|x_{t-1})"
        ).move_to(ORIGIN+0.3*DOWN)
        gauss_forward_ddpm_formula = MathTex(
            r"q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbb{I}) \\ \beta_t \in (0, 1) \text{ and } \beta_{t-1} < \beta_t"
        ).move_to(ORIGIN+2*DOWN)
        backward_ddpm_formula = MathTex(
            r"p(x_{0:T}) \equiv p(x_T)\prod_{t=1}^T p(x_{t-1}|x_t)"
        ).move_to(ORIGIN+0.3*DOWN)
        gauss_backward_ddpm_formula = MathTex(
            r"p_\theta(x_T) = \mathcal{N}(x_{T}; 0, \mathbb{I})"
        ).move_to(ORIGIN+2*DOWN)
        backward_parametrized_ddpm_formula = MathTex(
            r"p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t),\Sigma_\theta(x_t,t))"
        ).move_to(ORIGIN+2*DOWN)

        
        # Forward
        forward_title = Text("Forward", color=WHITE).to_edge(UP)
        backward_title = Text("Backward", color=WHITE).to_edge(UP)
        arrow_forward = Arrow(forward_title.get_left(), forward_title.get_right()).shift(DOWN)
        arrow_backward = Arrow(backward_title.get_right(), backward_title.get_left()).shift(DOWN)
        txt_x0.to_edge(ORIGIN + 3.5*LEFT)
        txt_x1 = MathTex(r"x_1").next_to(txt_x0, RIGHT, 2)
        txt_x2 = MathTex(r"x_2").next_to(txt_x1, RIGHT, 2)
        txt_x3 = MathTex(r"...").next_to(txt_x2, RIGHT, 2)
        txt_xT = MathTex(r"x_T").next_to(txt_x3, RIGHT, 2)
        arrow_txt_x1 = Arrow(txt_x0.get_right(), txt_x1.get_left())
        arrow_txt_x2 = Arrow(txt_x1.get_right(), txt_x2.get_left())
        arrow_txt_x3 = Arrow(txt_x2.get_right(), txt_x3.get_left())
        arrow_txt_xT = Arrow(txt_x3.get_right(), txt_xT.get_left())

        self.play(
            Write(forward_title),
            GrowArrow(arrow_forward)
        )
        self.wait(2)

        self.play(Write(txt_x0), run_time=0.5)
        self.play(GrowArrow(arrow_txt_x1), run_time=0.5)
        self.play(Write(txt_x1), run_time=0.5)
        self.play(GrowArrow(arrow_txt_x2), run_time=0.5)
        self.play(Write(txt_x2), run_time=0.5)
        self.play(GrowArrow(arrow_txt_x3), run_time=0.5)
        self.play(Write(txt_x3), run_time=0.5)
        self.play(GrowArrow(arrow_txt_xT), run_time=0.5)
        self.play(Write(txt_xT), run_time=0.5)
        self.wait(2)
        g_txt_x0_xT = Group(txt_x0, arrow_txt_x1, txt_x1, arrow_txt_x2, txt_x2, arrow_txt_x3, txt_x3, arrow_txt_xT, txt_xT)
        self.play(
            LaggedStart(
                AnimationGroup(
                    g_txt_x0_xT.animate.move_to(ORIGIN+UP),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        self.play(
            Write(forward_ddpm_formula)
        )
        self.wait(2)
        self.play(
            Write(gauss_forward_ddpm_formula)
        )
        self.wait(2)
        self.play(
            FadeOut(
                forward_title,
                arrow_forward,
                g_txt_x0_xT,
                forward_ddpm_formula,
                gauss_forward_ddpm_formula,
            )
        )

        # Backward
        txt_x0 = MathTex(r"x_0").to_edge(ORIGIN + 3.5*LEFT)
        txt_x1 = MathTex(r"x_1").next_to(txt_x0, RIGHT, 2)
        txt_x2 = MathTex(r"...").next_to(txt_x1, RIGHT, 2)
        txt_x3 = MathTex(r"x_{T-1}").next_to(txt_x2, RIGHT, 2)
        txt_xT = MathTex(r"x_T").next_to(txt_x3, RIGHT, 2)
        arrow_txt_x1 = Arrow(txt_x1.get_left(), txt_x0.get_right())
        arrow_txt_x2 = Arrow(txt_x2.get_left(), txt_x1.get_right())
        arrow_txt_x3 = Arrow(txt_x3.get_left(), txt_x2.get_right())
        arrow_txt_xT = Arrow(txt_xT.get_left(), txt_x3.get_right())

        self.play(
            Write(backward_title),
            GrowArrow(arrow_backward)
        )
        self.wait(2)
        
        self.play(Write(txt_xT), run_time=0.5)
        self.play(GrowArrow(arrow_txt_xT), run_time=0.5)
        self.play(Write(txt_x3), run_time=0.5)
        self.play(GrowArrow(arrow_txt_x3), run_time=0.5)
        self.play(Write(txt_x2), run_time=0.5)
        self.play(GrowArrow(arrow_txt_x2), run_time=0.5)
        self.play(Write(txt_x1), run_time=0.5)
        self.play(GrowArrow(arrow_txt_x1), run_time=0.5)
        self.play(Write(txt_x0), run_time=0.5)
        self.wait(2)
        g_txt_x0_xT = Group(txt_xT, arrow_txt_xT, txt_x3, arrow_txt_x3, txt_x2, arrow_txt_x2, txt_x1, arrow_txt_x1, txt_x0)
        self.play(
            LaggedStart(
                AnimationGroup(
                    g_txt_x0_xT.animate.move_to(ORIGIN+UP),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        self.play(
            Write(backward_ddpm_formula)
        )
        self.wait(2)
        self.play(
            Write(gauss_backward_ddpm_formula)
        )
        self.wait(2)

        # # VAE architecture
        # self.next_section(skip_animations=False)

        # encoder = Polygon(
        #     [-1, 1.4, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1.4, 0], color=PURPLE
        # )
        # encoder_txt = Text("Encoder", color=WHITE).scale(0.6).move_to(encoder)

        # mu = Rectangle(height=0.6, width=0.45, color=YELLOW)
        # mu_txt = MathTex(r"\mu").scale(0.8).move_to(mu)
        # sigma = Rectangle(height=0.6, width=0.45, color=YELLOW)
        # sigma_txt = MathTex(r"\sigma").scale(0.8).move_to(sigma)

        # mu = VGroup(mu, mu_txt)
        # sigma = VGroup(sigma, sigma_txt)
        # params = VGroup(mu, sigma).arrange(DOWN, buff=0.1).next_to(encoder, RIGHT, 0.25)

        # distrib = Rectangle(height=1.2, width=1.8, color=RED).next_to(
        #     params, RIGHT, buff=0.25
        # )
        # distrib_txt = MathTex(r"\mathcal{N}(\mu, \sigma)").scale(0.8).move_to(distrib)
        # distrib = VGroup(distrib, distrib_txt)

        # sample = MathTex(r"\sim").scale(0.8).next_to(distrib, RIGHT, buff=0.2)

        # z = Rectangle(height=1.2, width=0.45, color=BLUE).next_to(
        #     sample, RIGHT, buff=0.2
        # )
        # z_txt = MathTex(r"z").scale(0.8).move_to(z)
        # z = VGroup(z, z_txt)

        # decoder = Polygon(
        #     [-1, 0.6, 0], [1, 1.4, 0], [1, -1.4, 0], [-1, -0.6, 0], color=PURPLE
        # )
        # decoder.next_to(z, RIGHT, buff=0.2)
        # decoder_txt = Text("Decoder", color=WHITE).scale(0.6).move_to(decoder)

        # vae = VGroup(
        #     encoder, params, distrib, sample, z, decoder, encoder_txt, decoder_txt
        # ).move_to(ORIGIN)

        # vae_title = Text("Variational Autoencoder", color=WHITE).to_edge(UP)

        # self.play(Write(vae_title))
        # self.play(Create(vae), run_time=3)

        # self.play(
        #     ShowPassingFlash(
        #         SurroundingRectangle(vae, buff=0.4, color=WHITE), time_width=0.4
        #     ),
        #     run_time=2,
        # )

        # self.wait(0.6)

        # # Autoencoder architecture
        # self.next_section(skip_animations=False)

        # self.play(
        #     LaggedStart(
        #         FadeOut(distrib, distrib_txt, sample, params),
        #         AnimationGroup(
        #             z.animate.move_to(ORIGIN),
        #             VGroup(encoder, encoder_txt).animate.move_to(1.5 * LEFT),
        #             VGroup(decoder, decoder_txt).animate.move_to(1.5 * RIGHT),
        #         ),
        #         lag_ratio=0.5,
        #     ),
        #     run_time=2,
        # )aw

        # self.play(FadeOut(vae_title, z, encoder, decoder, encoder_txt, decoder_txt))

        # self.wait(2)


if __name__ == "__main__":
    scene = Scene1()
    scene.render()
