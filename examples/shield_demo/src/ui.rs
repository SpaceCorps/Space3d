//! The demo's panel (French or English), the ship labels and the floating "Miss" / "Immune".

use crate::catalog::*;
use crate::game::*;
use space3d::prelude::*;
use space3d::render::GraphicsQuality;

fn hull_name(g: &Game, i: usize) -> &'static str {
    g.tr(HULLS[i].fr, HULLS[i].en)
}

fn hull_combo(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    g: &Game,
    value: &mut usize,
    ships_only: bool,
    mobs_only: bool,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(id)
            .selected_text(hull_name(g, *value))
            .show_ui(ui, |ui| {
                for (i, h) in HULLS.iter().enumerate() {
                    let ship = h.kind == HullKind::Ship;
                    if (ships_only && !ship) || (mobs_only && ship) {
                        continue;
                    }
                    ui.selectable_value(value, i, g.tr(h.fr, h.en));
                }
            });
    });
}

fn slider(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str) -> bool {
    ui.add(egui::Slider::new(value, range).text(label)).changed()
}

fn color_row(ui: &mut egui::Ui, g: &Game, c: &mut [f32; 3]) {
    ui.horizontal_wrapped(|ui| {
        for (fr, en, p) in PALETTES {
            let col = egui::Color32::from_rgb(
                (p[0] / 2.8 * 255.0) as u8,
                (p[1] / 2.8 * 255.0) as u8,
                (p[2] / 2.8 * 255.0) as u8,
            );
            let b = egui::Button::new(egui::RichText::new("●").color(col)).selected(*c == p);
            if ui.add(b).on_hover_text(g.tr(fr, en)).clicked() {
                *c = p;
            }
        }
    });
    for (k, name) in ["R", "G", "B"].into_iter().enumerate() {
        ui.add(egui::Slider::new(&mut c[k], 0.0..=4.0).text(name));
    }
}

pub fn panel(ui: &egui::Context, ctx: &mut SystemCtx) {
    let Some(mut g) = ctx.resources.remove::<Game>() else {
        return;
    };
    draw_labels(ui, ctx, &g);
    let mut fire = false;
    let mut brk = false;
    let mut restore = false;
    let mut aegis = false;
    let mut pct_changed = false;
    let title = g.tr("Bouclier 3D — moteur Space3d", "3D shield — Space3d engine");
    let max_h = (ui.content_rect().height() - 40.0).max(120.0);
    let wide = ui.content_rect().width() > 760.0;
    egui::Window::new(title)
        .id(egui::Id::new("shield_panel"))
        .default_open(wide)
        .default_pos([12.0, 12.0])
        .default_width(330.0)
        .max_height(max_h)
        .resizable(false)
        .show(ui, |ui| {
            egui::ScrollArea::vertical().max_height(max_h - 40.0).show(ui, |ui| {
                controls(
                    ui,
                    &mut g,
                    &mut fire,
                    &mut brk,
                    &mut restore,
                    &mut aegis,
                    &mut pct_changed,
                );
            });
        });
    g.ui_busy = ui.egui_is_using_pointer()
        || ui
            .input(|i| i.pointer.interact_pos())
            .and_then(|p| ui.layer_id_at(p))
            .is_some_and(|l| l.order != egui::Order::Background);
    g.pointer(ctx);
    for (clicked, cmd) in [
        (fire, Cmd::Fire(None, None)),
        (brk, Cmd::Break),
        (restore, Cmd::Restore),
        (aegis, Cmd::Aegis),
        (pct_changed, Cmd::Pct),
    ] {
        if clicked {
            g.queue(cmd);
        }
    }
    ctx.resources.insert(g);
}

fn controls(
    ui: &mut egui::Ui,
    g: &mut Game,
    fire: &mut bool,
    brk: &mut bool,
    restore: &mut bool,
    aegis: &mut bool,
    pct_changed: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.selectable_value(&mut g.s.lang, Lang::Fr, "FR");
        ui.selectable_value(&mut g.s.lang, Lang::En, "EN");
        ui.separator();
        ui.label(format!("{:.0} {}", g.fps, g.tr("im/s", "fps")));
    });
    if let Some(text) = g.loading_text() {
        ui.colored_label(egui::Color32::LIGHT_YELLOW, text);
    }
    if g.fallback {
        ui.colored_label(
            egui::Color32::LIGHT_YELLOW,
            g.tr(
                "Modèles SpaceCorps absents : chaque vaisseau est le vaisseau de test du moteur (voir README).",
                "SpaceCorps models not found: every hull is the engine's test ship (see README).",
            ),
        );
    }
    ui.horizontal(|ui| {
        if ui.button(g.tr("Tirer (Espace)", "Fire (Space)")).clicked() {
            *fire = true;
        }
        let pause = if g.s.paused {
            g.tr("Reprendre", "Resume")
        } else {
            g.tr("Pause (P)", "Pause (P)")
        };
        if ui.button(pause).clicked() {
            g.s.paused = !g.s.paused;
        }
    });
    ui.horizontal(|ui| {
        ui.label(g.tr("Vitesse", "Speed"));
        for (v, l) in [(1.0, "1×"), (0.5, "0.5×"), (0.25, "0.25×"), (0.1, "0.1×")] {
            ui.selectable_value(&mut g.s.speed, v, l);
        }
    });
    ui.horizontal(|ui| {
        ui.label(g.tr("Cible", "Target"));
        let (all, you, player) = (g.tr("Tous", "All"), g.tr("Vous", "You"), g.tr("Joueur", "Player"));
        ui.selectable_value(&mut g.s.target, Target::All, all);
        ui.selectable_value(&mut g.s.target, Target::You, you);
        ui.add_enabled_ui(g.s.multi, |ui| {
            ui.selectable_value(&mut g.s.target, Target::Player, player);
            ui.selectable_value(&mut g.s.target, Target::Mob, "Mob");
        });
    });
    ui.horizontal(|ui| {
        ui.label(g.tr("Qualité", "Quality"));
        let tiers = [
            (GraphicsQuality::Low, g.tr("Faible", "Low")),
            (GraphicsQuality::Medium, g.tr("Moyen", "Medium")),
            (GraphicsQuality::High, g.tr("Élevé", "High")),
            (GraphicsQuality::Ultra, "Ultra"),
        ];
        for (q, l) in tiers {
            if ui.selectable_label(g.s.quality == q, l).clicked() {
                g.set_quality(q);
            }
        }
    });
    let (auto, rate, sound, vol) = (
        g.tr("Tir auto", "Auto fire"),
        g.tr("Cadence (salves/s)", "Fire rate (volleys/s)"),
        g.tr("Son", "Sound"),
        g.tr("Volume", "Volume"),
    );
    ui.horizontal(|ui| {
        ui.checkbox(&mut g.s.auto, auto);
        ui.checkbox(&mut g.s.sound, sound);
    });
    slider(ui, &mut g.s.rate, 0.5..=8.0, rate);
    if matches!(g.weapon().kind, WeaponKind::Rocket | WeaponKind::Blast) {
        ui.weak(format!("{:.2}{}", g.s.rate / 3.0, g.tr(" roquette/s", " rocket/s")));
    }
    slider(ui, &mut g.s.volume, 0.0..=1.0, vol);

    // scene
    egui::CollapsingHeader::new(g.tr("Scène", "Scene"))
        .id_salt("scene")
        .default_open(true)
        .show(ui, |ui| {
            let mut v = g.s.main;
            hull_combo(ui, "main", g.tr("Modèle", "Model"), g, &mut v, false, false);
            g.s.main = v;
            ui.horizontal(|ui| {
                let (solo, multi) = (g.tr("Seul", "Solo"), g.tr("À plusieurs", "Multiplayer"));
                ui.selectable_value(&mut g.s.multi, false, solo);
                ui.selectable_value(&mut g.s.multi, true, multi);
            });
            if g.s.multi {
                let mut v = g.s.ally;
                hull_combo(ui, "ally", g.tr("Autre joueur", "Other player"), g, &mut v, true, false);
                g.s.ally = v;
                ui.horizontal(|ui| {
                    ui.label(g.tr("Son skin", "Their skin"));
                    let current = PALETTES[g.s.ally_skin];
                    let text = g.tr(current.0, current.1);
                    let names: Vec<&'static str> = PALETTES.iter().map(|p| g.tr(p.0, p.1)).collect();
                    egui::ComboBox::from_id_salt("ally_skin")
                        .selected_text(text)
                        .show_ui(ui, |ui| {
                            for (i, n) in names.iter().enumerate() {
                                ui.selectable_value(&mut g.s.ally_skin, i, *n);
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label(g.tr("Son absorption", "Their absorbance"));
                    for a in [45.0, 62.0, 91.0, 109.0] {
                        ui.selectable_value(&mut g.s.ally_absorb, a, format!("{a:.0}"));
                    }
                });
                let mut v = g.s.foe;
                hull_combo(ui, "foe", "Mob", g, &mut v, false, true);
                g.s.foe = v;
            }
            ui.horizontal(|ui| {
                let (game, orbit) = (g.tr("Vue de jeu", "Game view"), g.tr("Orbite", "Orbit"));
                ui.selectable_value(&mut g.s.orbit, false, game);
                ui.selectable_value(&mut g.s.orbit, true, orbit);
                let spin = g.tr("Rotation lente", "Slow rotation");
                ui.checkbox(&mut g.s.spin, spin);
            });
        });

    // shots
    egui::CollapsingHeader::new(g.tr("Tirs", "Shots"))
        .id_salt("shots")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(g.tr("Arme", "Weapon"));
                let current = g.weapon();
                let text = g.tr(current.fr, current.en);
                let names: Vec<&'static str> = WEAPONS.iter().map(|w| g.tr(w.fr, w.en)).collect();
                egui::ComboBox::from_id_salt("weapon")
                    .selected_text(text)
                    .show_ui(ui, |ui| {
                        for (i, n) in names.iter().enumerate() {
                            ui.selectable_value(&mut g.s.weapon, i, *n);
                        }
                    });
            });
            if g.weapon().pen.is_none() {
                let t = g.tr("Pénétration (%)", "Penetration (%)");
                slider(ui, &mut g.s.pen, 0.0..=105.0, t);
            }
            ui.horizontal(|ui| {
                ui.label(g.tr("Tireur", "Shooter"));
                egui::ComboBox::from_id_salt("shooter")
                    .selected_text(GUNS[g.s.shooter].label)
                    .show_ui(ui, |ui| {
                        for (i, c) in GUNS.iter().enumerate() {
                            ui.selectable_value(&mut g.s.shooter, i, c.label);
                        }
                    });
            });
            ui.horizontal(|ui| {
                let (own, other) = (g.tr("Vos tirs", "Your shots"), g.tr("Un autre pilote", "Another pilot"));
                ui.selectable_value(&mut g.s.view_self, true, own);
                ui.selectable_value(&mut g.s.view_self, false, other);
            });
            ui.horizontal(|ui| {
                ui.label(g.tr("Tirs par salve", "Bolts per volley"));
                ui.add_enabled_ui(g.s.view_self, |ui| {
                    ui.selectable_value(&mut g.s.shots, 0, "auto");
                    for n in [2, 3, 4] {
                        ui.selectable_value(&mut g.s.shots, n, n.to_string());
                    }
                });
            });
            let miss = g.tr("Tirs ratés (20 %)", "Missed shots (20%)");
            ui.checkbox(&mut g.s.miss, miss);
            let fx = g.tr("Éclats quand un tir perfore", "Fragments when a shot pierces");
            ui.checkbox(&mut g.s.piercing_fx, fx);
            ui.small(split_text(g));
        });

    // shield state
    egui::CollapsingHeader::new(g.tr("État du bouclier", "Shield status"))
        .id_salt("status")
        .default_open(true)
        .show(ui, |ui| {
            let t = g.tr("Bouclier (%)", "Shield (%)");
            if slider(ui, &mut g.s.pct, 0.0..=100.0, t) {
                *pct_changed = true;
            }
            let wear = g.tr("Les tirs usent le bouclier", "Shots wear the shield");
            ui.checkbox(&mut g.s.wear, wear);
            ui.add_enabled_ui(g.s.wear, |ui| {
                ui.horizontal(|ui| {
                    ui.label(g.tr("Recharge après", "Regenerates after"));
                    ui.selectable_value(&mut g.s.regen, 15.0, "15 s");
                    ui.selectable_value(&mut g.s.regen, 3.0, "3 s");
                });
            });
            ui.add_enabled_ui(g.is_ship(g.s.main), |ui| {
                let t = g.tr("Votre absorption (%)", "Your absorbance (%)");
                slider(ui, &mut g.s.absorb, 45.0..=160.0, t);
            });
            ui.horizontal(|ui| {
                if ui.button(g.tr("Casser (B)", "Break (B)")).clicked() {
                    *brk = true;
                }
                if ui.button(g.tr("Restaurer (R)", "Restore (R)")).clicked() {
                    *restore = true;
                }
            });
        });

    // Aegis
    egui::CollapsingHeader::new(g.tr("Égide : bouclier de groupe", "Aegis: group shield")).id_salt("aegis").default_open(true).show(ui, |ui| {
        ui.small(g.tr(
            "Votre bouclier s'étend en une bulle qui vous couvre, vous et vos alliés, pendant 8 s. Votre coque n'a plus de bouclier pendant ce temps.",
            "Your shield stretches into a bubble covering you and your allies for 8 s. Your hull has no shield meanwhile.",
        ));
        if ui.button(g.tr("Déployer l'Égide (F)", "Deploy Aegis (F)")).clicked() {
            *aegis = true;
        }
        ui.horizontal(|ui| {
            ui.label(g.tr("Recharge", "Cooldown"));
            let (demo, game) = (g.tr("5 s (démo)", "5 s (demo)"), g.tr("120 s (jeu)", "120 s (game)"));
            ui.selectable_value(&mut g.s.aegis_cooldown, 5.0, demo);
            ui.selectable_value(&mut g.s.aegis_cooldown, 120.0, game);
        });
        ui.small(g.aegis_text());
    });

    // colours
    egui::CollapsingHeader::new(g.tr("Couleur", "Color"))
        .id_salt("color")
        .show(ui, |ui| {
            if g.is_ship(g.s.main) {
                ui.label(g.tr("Votre skin de bouclier", "Your shield skin"));
                let mut c = g.s.skin;
                color_row(ui, g, &mut c);
                g.s.skin = c;
            } else {
                ui.label(g.tr("Couleur de ce type de mob", "Color of this mob type"));
                let mut c = g.color_of_hull(g.s.main);
                let before = c;
                color_row(ui, g, &mut c);
                if c != before {
                    g.s.mob_colors.insert(g.s.main, c);
                }
                if ui.button(g.tr("Couleur par défaut", "Default color")).clicked() {
                    g.s.mob_colors.remove(&g.s.main);
                }
            }
        });

    // look
    egui::CollapsingHeader::new(g.tr("Rendu du bouclier", "Shield look"))
        .id_salt("look")
        .show(ui, |ui| {
            let names = [
                g.tr("Intensité", "Intensity"),
                g.tr("Lueur au repos", "Rest glow"),
                g.tr("Bord (fresnel)", "Rim (fresnel)"),
                g.tr("Taille des hexagones", "Hex size"),
                g.tr("Vitesse des ondes", "Wave speed"),
                g.tr("Largeur des ondes", "Wave width"),
                g.tr("Durée des ondes", "Wave life"),
                g.tr("Éclair d'impact", "Impact flash"),
                "Bloom",
                g.tr("Exposition (EV)", "Exposure (EV)"),
            ];
            slider(ui, &mut g.s.intensity, 0.2..=3.0, names[0]);
            slider(ui, &mut g.s.rest, 0.0..=1.5, names[1]);
            slider(ui, &mut g.s.fresnel, 0.5..=8.0, names[2]);
            let hex = g.tr("Hexagones", "Hexagons");
            ui.checkbox(&mut g.s.hex, hex);
            slider(ui, &mut g.s.hex_scale, 4.0..=40.0, names[3]);
            slider(ui, &mut g.s.wave_speed, 0.4..=4.0, names[4]);
            slider(ui, &mut g.s.wave_width, 0.02..=0.2, names[5]);
            slider(ui, &mut g.s.wave_life, 0.3..=2.0, names[6]);
            slider(ui, &mut g.s.flash, 0.0..=3.0, names[7]);
            slider(ui, &mut g.s.bloom, 0.0..=1.5, names[8]);
            slider(ui, &mut g.s.exposure, -2.0..=2.0, names[9]);
        });
}

fn pc(v: f32) -> String {
    format!("{:.0}", v * 100.0)
}

/// What the chosen weapon does to each ship in the scene.
fn split_text(g: &Game) -> String {
    let w = g.weapon();
    let fr = g.s.lang == Lang::Fr;
    let mut lines = Vec::new();
    if w.kind == WeaponKind::Laser {
        let n = g.volley_count();
        let guns = match (n, fr) {
            (2, true) => "depuis la paire de canons principale",
            (2, false) => "from the main pair of guns",
            (3, true) => "depuis la paire principale et le canon central",
            (3, false) => "from the main pair and the center gun",
            (_, true) => "depuis les deux paires de canons",
            (_, false) => "from both pairs of guns",
        };
        lines.push(if fr {
            format!("Salve : {n} tirs, {guns} ({}).", g.guns().label)
        } else {
            format!("Volley: {n} shots, {guns} ({}).", g.guns().label)
        });
    } else if fr {
        lines.push("Un seul tir, du centre du tireur.".to_string());
    } else {
        lines.push("A single shot, from the shooter's center.".to_string());
    }
    if w.kind == WeaponKind::Siphon && g.s.view_self && g.units.len() > 1 {
        lines.push(
            g.tr(
                "Votre Siphon sur l'autre joueur ou le mob part de votre vaisseau. Le bouclier pris \
                 revient en paquets : le vôtre gagne 7 % (avec l'usure, ou dans la réserve de \
                 l'Égide), et chaque paquet y lance une onde.",
                "Your Siphon at the other player or the mob leaves from your ship. The shield it \
                 takes comes back as packets: yours takes in 7% (with wear, or into the Aegis \
                 reserve), and each packet sends a wave over it.",
            )
            .to_string(),
        );
    }
    for (i, u) in g.units.iter().enumerate() {
        let name = match u.role {
            Role::Main => g.tr("Vous", "You").to_string(),
            Role::Ally => g.tr("Autre joueur", "Other player").to_string(),
            Role::Foe => hull_name(g, u.hull).to_string(),
        };
        let a = g.absorb_of(i);
        let text = if g.immune(i) {
            g.tr("tirs sans effet", "shots have no effect").to_string()
        } else if w.kind == WeaponKind::Siphon {
            g.tr("bouclier seul", "shield only").to_string()
        } else {
            let s = g.share_of(a, w);
            if fr {
                format!("bouclier {} % · coque {} %", pc(s), pc(1.0 - s))
            } else {
                format!("shield {}% · hull {}%", pc(s), pc(1.0 - s))
            }
        };
        let pct = if g.immune(i) {
            String::new()
        } else {
            format!(" ({a:.0}{})", if fr { " %" } else { "%" })
        };
        lines.push(format!("{name}{pct}{} {text}", if fr { " :" } else { ":" }));
    }
    lines.join("\n")
}

/// Ship labels and floating texts, drawn behind the windows.
fn draw_labels(ui: &egui::Context, ctx: &SystemCtx, g: &Game) {
    let rect = ui.content_rect();
    let Some(vp) = g.view_proj(ctx, rect.width() / rect.height().max(1.0)) else {
        return;
    };
    let painter = ui.layer_painter(egui::LayerId::background());
    let project = |p: Vec3| {
        let c = vp * p.extend(1.0);
        (c.w > 0.0).then(|| {
            let n = c.truncate() / c.w;
            egui::pos2(
                rect.left() + (n.x * 0.5 + 0.5) * rect.width(),
                rect.top() + (0.5 - n.y * 0.5) * rect.height(),
            )
        })
    };
    let fr = g.s.lang == Lang::Fr;
    let aegis = g.aegis_up();
    for (i, u) in g.units.iter().enumerate() {
        let Some(at) = project(u.pos + Vec3::new(0.0, u.bound() * 0.4, -u.bound() * 1.05)) else {
            continue;
        };
        let aimed = match (g.s.target, u.role) {
            (Target::You, Role::Main) | (Target::Player, Role::Ally) | (Target::Mob, Role::Foe) => "◎ ",
            _ => "",
        };
        let who = match u.role {
            Role::Main => g.tr("Vous", "You"),
            Role::Ally => g.tr("Joueur", "Player"),
            Role::Foe => "Mob",
        };
        let mut text = format!("{aimed}{who} · {}", hull_name(g, u.hull));
        if u.role == Role::Main {
            let state = match g.state {
                State::Down => g.tr("bouclier tombé", "shield down").to_string(),
                State::Breaking => g.tr("rupture", "breaking").to_string(),
                State::Restoring => g.tr("réapparition", "coming back").to_string(),
                State::Up => format!("{:.0}{}", g.s.pct, if fr { " %" } else { "%" }),
            };
            text.push_str(&format!(" · {state}"));
        }
        if u.role == Role::Ally && aegis {
            text.push_str(if fr { " · protégé" } else { " · protected" });
        }
        let c = g.unit_color(i);
        let m = c[0].max(c[1]).max(c[2]).max(1e-3);
        let col = egui::Color32::from_rgb(
            (120.0 + 135.0 * c[0] / m) as u8,
            (120.0 + 135.0 * c[1] / m) as u8,
            (120.0 + 135.0 * c[2] / m) as u8,
        );
        painter.text(
            at,
            egui::Align2::CENTER_BOTTOM,
            text,
            egui::FontId::proportional(13.0),
            col,
        );
    }
    for f in &g.floats {
        let t = ((g.sim - f.t) / 0.95).clamp(0.0, 1.0);
        let Some(at) = project(f.pos + Vec3::Y * (g.world_len * 0.3 * t)) else {
            continue;
        };
        let col = egui::Color32::from_white_alpha(((1.0 - t) * 255.0) as u8);
        painter.text(
            at,
            egui::Align2::CENTER_CENTER,
            f.text,
            egui::FontId::proportional(18.0),
            col,
        );
    }
}
