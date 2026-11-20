//! Inspector panel: typed editors for `Name`, `Transform`, `Parent`, `Material`, lights,
//! `Camera3d`, `Visibility`, `MeshSource`, `Mesh3d`, the graphics components (`CameraFrame`,
//! `PostProcess`, `Skybox`, `EnvironmentLight`, `DirectionalShadow`), plus a JSON editor for
//! every other component registered in the scene `ComponentRegistry`. All edits are undoable.

use crate::components::{
    ANTI_ALIASING, HDR_MODES, MeshSource, SHADOW_FILTERS, TONEMAPPERS, can_remove, component_values, ensure_stable_id,
    names, remove_component, set_component_value,
};
use crate::history::{EditCommand, EditorHistory};
use crate::widgets::{
    fill_drag, fill_slider, icon_button, number_field, property, property_group, section, vec3_fields,
};
use space3d_core::glam::{EulerRot, Quat, Vec3, Vec4};
use space3d_core::hecs::{Component, Entity, World};
use space3d_core::{Name, Parent, Transform};
use space3d_render::{
    Camera3d, CameraFrame, DirectionalLight, DirectionalShadow, EnvironmentLight, Material, Mesh3d, PointLight,
    PostProcess, Projection, RenderAssets, RenderTarget, Skybox, Visibility,
};
use space3d_scene::{ComponentRegistry, ComponentValue, StableId};
use std::collections::{BTreeMap, HashMap};

/// Per-selection inspector state.
#[derive(Debug, Default)]
pub struct InspectorState {
    /// JSON text being edited, by component name.
    json_edits: HashMap<String, String>,
    /// Last JSON error by component name.
    json_errors: HashMap<String, String>,
    /// Euler angles shown for the rotation (avoids quaternion → euler jitter while dragging).
    euler: Option<(Entity, Quat, Vec3)>,
    /// Component name picked in "Add component".
    add_pick: String,
    entity: Option<Entity>,
}

/// Handled by typed editors (not shown in the JSON section).
const TYPED: [&str; 12] = [
    "Name",
    space3d_scene::TRANSFORM,
    space3d_scene::PARENT,
    names::MATERIAL,
    names::DIRECTIONAL_LIGHT,
    names::POINT_LIGHT,
    names::CAMERA,
    names::VISIBILITY,
    names::MESH_SOURCE,
    names::CAMERA_FRAME,
    names::POST_PROCESS,
    names::DIRECTIONAL_SHADOW,
];

/// Not offered in "Add component".
const HIDDEN: [&str; 3] = ["StableId", "GlobalTransform", "Mesh3d"];

/// Did an edit gesture end (drag released, text field left, click)?
fn gesture_ended(r: &egui::Response) -> bool {
    r.drag_stopped() || r.lost_focus() || (r.changed() && !r.dragged() && !r.has_focus())
}

// egui's color buttons round-trip the value through HSVA every frame; writing that back
// unconditionally would nudge colors (and record bogus undo steps) just by showing them.
fn color3_edit(ui: &mut egui::Ui, c: &mut Vec3) -> egui::Response {
    let mut rgb = c.to_array();
    let r = ui.color_edit_button_rgb(&mut rgb);
    if r.changed() {
        *c = Vec3::from(rgb);
    }
    r
}

/// A [`property`] row choosing from a `(value, name)` table with a combo box filling the value
/// column (the selected name truncated to it); returns the response of the box.
fn choice_edit<T: PartialEq + Copy>(ui: &mut egui::Ui, label: &str, v: &mut T, table: &[(T, &str)]) -> egui::Response {
    property(ui, label, |ui| {
        let current = table.iter().find(|(t, _)| *t == *v).map_or("(other)", |(_, n)| n);
        let mut r = egui::ComboBox::from_id_salt(("space3d_inspector_choice", label))
            .width(ui.available_width())
            .truncate()
            .selected_text(current)
            .show_ui(ui, |ui| {
                let mut changed: Option<egui::Response> = None;
                for (t, n) in table {
                    let item = ui.selectable_label(*t == *v, *n);
                    if item.clicked() {
                        *v = *t;
                    }
                    changed = Some(match changed {
                        Some(c) => c | item,
                        None => item,
                    });
                }
                changed
            });
        if let Some(Some(inner)) = r.inner.take() {
            r.response |= inner;
        }
        r.response
    })
}

/// A "Yaw °" [`property`] row: the yaw (degrees about +Y) of a rotation, edited as a slider.
fn yaw_edit(ui: &mut egui::Ui, q: &mut Quat) -> egui::Response {
    let (yaw, _, _) = q.to_euler(EulerRot::YXZ);
    let mut deg = yaw.to_degrees();
    let r = property(ui, "Yaw °", |ui| {
        fill_slider(ui, egui::Slider::new(&mut deg, -180.0..=180.0))
    });
    if r.changed() {
        *q = Quat::from_rotation_y(deg.to_radians());
    }
    r
}

/// A [`property`] row with an on/off checkbox followed by a slider filling the rest of the line.
fn toggle_slider(ui: &mut egui::Ui, label: &str, on: &mut bool, slider: egui::Slider<'_>) -> egui::Response {
    property(ui, label, |ui| {
        let r = ui.checkbox(on, "");
        r | fill_slider(ui, slider)
    })
}

/// A [`property`] row with a drag value filling the value column.
fn drag_row(ui: &mut egui::Ui, label: &str, drag: egui::DragValue<'_>) -> egui::Response {
    property(ui, label, |ui| fill_drag(ui, drag))
}

fn color4_edit(ui: &mut egui::Ui, c: &mut Vec4) -> egui::Response {
    let mut rgba = egui::Rgba::from_rgba_unmultiplied(c.x, c.y, c.z, c.w);
    let r = egui::color_picker::color_edit_button_rgba(ui, &mut rgba, egui::color_picker::Alpha::OnlyBlend);
    if r.changed() {
        *c = Vec4::from(rgba.to_rgba_unmultiplied());
    }
    r
}

/// Collapsible typed editor. Writes changes to the world and records merged undo steps.
#[allow(clippy::too_many_arguments)]
fn typed_section<T: Component + Clone + PartialEq>(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &ComponentRegistry,
    history: &mut EditorHistory,
    e: Entity,
    id: StableId,
    title: &str,
    body: impl FnOnce(&mut egui::Ui, &mut T) -> Option<egui::Response>,
) -> bool {
    let Ok(before) = world.get::<&T>(e).map(|c| (*c).clone()) else {
        return false;
    };
    let mut value = before.clone();
    let mut ended = false;
    let (remove, _) = section(
        ui,
        ("space3d_inspector", title),
        title,
        true,
        |ui| icon_button(ui, "×", &format!("Remove {title}")).clicked(),
        |ui| {
            if let Some(r) = body(ui, &mut value) {
                ended = gesture_ended(&r);
            }
        },
    );
    if remove {
        history.execute(
            EditCommand::set::<T>(format!("Remove {title}"), id, Some(before), None),
            world,
            registry,
        );
        return true;
    }
    let changed = value != before;
    if changed {
        let _ = world.insert_one(e, value.clone());
        history.push(
            EditCommand::set(format!("Edit {title}"), id, Some(before), Some(value)).merge_key(format!("{title}:{id}")),
        );
    }
    if ended {
        history.end_merge();
    }
    changed
}

/// Draw the inspector for `e`. Returns `true` if a `MeshSource` changed (mesh needs rebuild).
pub fn inspector_ui(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &ComponentRegistry,
    assets: Option<&RenderAssets>,
    history: &mut EditorHistory,
    state: &mut InspectorState,
    e: Entity,
) -> bool {
    if state.entity != Some(e) {
        state.json_edits.clear();
        state.json_errors.clear();
        state.entity = Some(e);
        history.end_merge();
    }
    let Some(id) = ensure_stable_id(world, e) else {
        ui.label("entity no longer exists");
        return false;
    };
    // Name, id and parent.
    let before = world.get::<&Name>(e).ok().map(|n| n.0.clone());
    let mut text = before.clone().unwrap_or_default();
    let r = property(ui, "Name", |ui| {
        ui.add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY))
    });
    if r.changed() {
        let after = (!text.is_empty()).then(|| Name(text.clone()));
        match &after {
            Some(n) => {
                let _ = world.insert_one(e, n.clone());
            }
            None => {
                let _ = world.remove_one::<Name>(e);
            }
        }
        history.push(EditCommand::set("Rename", id, before.map(Name), after).merge_key(format!("name:{id}")));
    }
    if r.lost_focus() {
        history.end_merge();
    }
    property(ui, "Entity", |ui| {
        ui.label(format!("#{}", e.id()));
        ui.add(egui::Label::new(egui::RichText::new(id.to_string()).small()).truncate())
            .on_hover_text(id.to_string());
    });
    if let Ok(p) = world.get::<&Parent>(e).map(|p| p.0) {
        let pname = crate::components::display_name(world, p);
        let unparent = property(ui, "Parent", |ui| {
            let clicked = ui.button("Unparent").clicked();
            ui.add(egui::Label::new(pname).truncate());
            clicked
        });
        if unparent {
            let pid = ensure_stable_id(world, p);
            history.execute(EditCommand::reparent(id, pid, None), world, registry);
        }
    }
    ui.add_space(2.0);

    typed_section::<Transform>(ui, world, registry, history, e, id, "Transform", |ui, t| {
        let mut r = property(ui, "Position", |ui| vec3_fields(ui, &mut t.translation, 0.01));
        let euler = match state.euler {
            Some((ent, q, eu)) if ent == e && q == t.rotation => eu,
            _ => {
                let (x, y, z) = t.rotation.to_euler(EulerRot::XYZ);
                Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
            }
        };
        let mut eu = euler;
        let rr = property(ui, "Rotation °", |ui| vec3_fields(ui, &mut eu, 0.5));
        if eu != euler {
            t.rotation = Quat::from_euler(EulerRot::XYZ, eu.x.to_radians(), eu.y.to_radians(), eu.z.to_radians());
        }
        state.euler = Some((e, t.rotation, eu));
        r |= rr;
        r |= property(ui, "Scale", |ui| vec3_fields(ui, &mut t.scale, 0.01));
        Some(r)
    });

    typed_section::<Material>(ui, world, registry, history, e, id, "Material", |ui, m| {
        let mut r = property(ui, "Base color", |ui| color4_edit(ui, &mut m.base_color));
        r |= property(ui, "Metallic", |ui| {
            fill_slider(ui, egui::Slider::new(&mut m.metallic, 0.0..=1.0))
        });
        r |= property(ui, "Roughness", |ui| {
            fill_slider(ui, egui::Slider::new(&mut m.roughness, 0.0..=1.0))
        });
        r |= property(ui, "Emissive", |ui| color3_edit(ui, &mut m.emissive));
        r |= property_group(ui, "Flags", |ui| {
            ui.checkbox(&mut m.unlit, "Unlit")
                | ui.checkbox(&mut m.double_sided, "2-sided")
                | ui.checkbox(&mut m.alpha_blend, "Blend")
        });
        if let Some(t) = m.base_color_texture {
            property(ui, "Texture", |ui| {
                ui.label(format!("#{}", t.index()));
                if ui.button("Clear").clicked() {
                    m.base_color_texture = None;
                }
            });
        }
        Some(r)
    });

    typed_section::<DirectionalLight>(ui, world, registry, history, e, id, "Directional light", |ui, l| {
        let mut r = property(ui, "Color", |ui| color3_edit(ui, &mut l.color));
        r |= property(ui, "Intensity", |ui| {
            fill_drag(
                ui,
                egui::DragValue::new(&mut l.intensity).speed(0.05).range(0.0..=1000.0),
            )
        });
        Some(r)
    });

    typed_section::<PointLight>(ui, world, registry, history, e, id, "Point light", |ui, l| {
        let mut r = property(ui, "Color", |ui| color3_edit(ui, &mut l.color));
        r |= property(ui, "Intensity", |ui| {
            fill_drag(
                ui,
                egui::DragValue::new(&mut l.intensity).speed(0.05).range(0.0..=1000.0),
            )
        });
        r |= property(ui, "Range", |ui| {
            fill_drag(ui, egui::DragValue::new(&mut l.range).speed(0.1).range(0.01..=10_000.0))
        });
        Some(r)
    });

    typed_section::<Camera3d>(ui, world, registry, history, e, id, "Camera", |ui, c| {
        let mut ortho = matches!(c.projection, Projection::Orthographic { .. });
        let mut r = property_group(ui, "Projection", |ui| {
            ui.selectable_value(&mut ortho, false, "Perspective") | ui.selectable_value(&mut ortho, true, "Ortho")
        });
        c.projection = match (ortho, c.projection) {
            (true, Projection::Perspective { near, far, .. }) => Projection::Orthographic {
                height: 10.0,
                near,
                far,
            },
            (false, Projection::Orthographic { near, far, .. }) => Projection::Perspective {
                fov_y: 60f32.to_radians(),
                near,
                far,
            },
            (_, p) => p,
        };
        match &mut c.projection {
            Projection::Perspective { fov_y, near, far } => {
                let mut deg = fov_y.to_degrees();
                r |= property(ui, "FOV °", |ui| {
                    fill_slider(ui, egui::Slider::new(&mut deg, 5.0..=170.0))
                });
                *fov_y = deg.to_radians();
                r |= property(ui, "Near / far", |ui| {
                    pair(
                        ui,
                        egui::DragValue::new(near).speed(0.01).range(0.0001..=100.0),
                        egui::DragValue::new(far).speed(1.0).range(0.01..=1e6),
                        0,
                    )
                });
            }
            Projection::Orthographic { height, near, far } => {
                r |= property(ui, "Height", |ui| {
                    fill_drag(ui, egui::DragValue::new(height).speed(0.1).range(0.01..=1e5))
                });
                r |= property(ui, "Near / far", |ui| {
                    pair(
                        ui,
                        egui::DragValue::new(near).speed(0.01),
                        egui::DragValue::new(far).speed(1.0),
                        0,
                    )
                });
            }
        }
        let mut clear = c.clear_color.is_some();
        r |= property(ui, "Clear", |ui| {
            let mut r = ui.checkbox(&mut clear, "");
            if let Some(col) = &mut c.clear_color
                && clear
            {
                r |= color4_edit(ui, col);
            }
            r
        });
        match (clear, &mut c.clear_color) {
            (true, None) => c.clear_color = Some(Vec4::new(0.02, 0.02, 0.03, 1.0)),
            (false, _) => c.clear_color = None,
            _ => {}
        }
        r |= property(ui, "Order", |ui| fill_drag(ui, egui::DragValue::new(&mut c.order)));
        r |= property(ui, "Active", |ui| ui.checkbox(&mut c.active, ""));
        if let RenderTarget::Texture(t) = c.target {
            property(ui, "Target", |ui| ui.label(format!("texture #{}", t.index())));
        }
        Some(r)
    });

    typed_section::<CameraFrame>(ui, world, registry, history, e, id, "Camera frame", |ui, f| {
        let mut r = choice_edit(ui, "HDR", &mut f.hdr, &HDR_MODES);
        let mut own = f.render_scale.is_some();
        r |= property(ui, "Render scale", |ui| {
            let mut r = ui.checkbox(&mut own, "").on_hover_text("own render scale");
            if let (true, Some(scale)) = (own, &mut f.render_scale) {
                r |= fill_slider(ui, egui::Slider::new(scale, 0.25..=1.0));
            }
            r
        });
        match (own, f.render_scale) {
            (true, None) => f.render_scale = Some(1.0),
            (false, Some(_)) => f.render_scale = None,
            _ => {}
        }
        Some(r)
    });

    typed_section::<PostProcess>(ui, world, registry, history, e, id, "Post-processing", |ui, p| {
        let mut r = property(ui, "Enabled", |ui| ui.checkbox(&mut p.enabled, ""));
        r |= drag_row(
            ui,
            "Exposure EV",
            egui::DragValue::new(&mut p.exposure).speed(0.05).range(-16.0..=16.0),
        );
        r |= property(ui, "Auto exposure", |ui| ui.checkbox(&mut p.auto_exposure.enabled, ""));
        r |= choice_edit(ui, "Tonemapper", &mut p.tonemapper, &TONEMAPPERS);
        r |= toggle_slider(
            ui,
            "Bloom",
            &mut p.bloom.enabled,
            egui::Slider::new(&mut p.bloom.intensity, 0.0..=2.0),
        );
        r |= property(ui, "Bloom threshold", |ui| {
            fill_slider(ui, egui::Slider::new(&mut p.bloom.threshold, 0.0..=8.0))
        });
        r |= choice_edit(ui, "Anti-aliasing", &mut p.anti_aliasing, &ANTI_ALIASING);
        r |= toggle_slider(
            ui,
            "Vignette",
            &mut p.vignette.enabled,
            egui::Slider::new(&mut p.vignette.intensity, 0.0..=1.0),
        );
        // One slider: 0 turns the effect off.
        let mut ca = if p.chromatic_aberration.enabled {
            p.chromatic_aberration.intensity
        } else {
            0.0
        };
        let ca_r = property(ui, "Chromatic aberration", |ui| {
            fill_slider(ui, egui::Slider::new(&mut ca, 0.0..=1.0))
        });
        if ca_r.changed() {
            p.chromatic_aberration.enabled = ca > 0.0;
            p.chromatic_aberration.intensity = ca;
        }
        r |= ca_r;
        r |= toggle_slider(
            ui,
            "Film grain",
            &mut p.film_grain.enabled,
            egui::Slider::new(&mut p.film_grain.intensity, 0.0..=0.5),
        );
        r |= toggle_slider(
            ui,
            "Distortion",
            &mut p.distortion.enabled,
            egui::Slider::new(&mut p.distortion.strength, 0.0..=4.0),
        );
        r |= property(ui, "Saturation", |ui| {
            fill_slider(ui, egui::Slider::new(&mut p.color_grading.saturation, 0.0..=2.0))
        });
        r |= property(ui, "Contrast", |ui| {
            fill_slider(ui, egui::Slider::new(&mut p.color_grading.contrast, 0.5..=2.0))
        });
        r |= property(ui, "Temperature", |ui| {
            fill_slider(ui, egui::Slider::new(&mut p.color_grading.temperature, -1.0..=1.0))
        });
        r |= property(ui, "Display gamma", |ui| {
            fill_slider(ui, egui::Slider::new(&mut p.display_gamma, 0.5..=2.0))
        });
        r |= property(ui, "Dither", |ui| ui.checkbox(&mut p.dither, ""));
        if p.color_grading.lut.is_some() {
            property(ui, "LUT", |ui| {
                ui.add(egui::Label::new("3D LUT (session-local: not saved)").truncate())
            });
        }
        Some(r)
    });

    typed_section::<Skybox>(ui, world, registry, history, e, id, "Skybox", |ui, sky| {
        property(ui, "Cubemap", |ui| ui.label(format!("#{}", sky.cubemap.index())));
        let mut r = drag_row(
            ui,
            "Intensity",
            egui::DragValue::new(&mut sky.intensity).speed(0.01).range(0.0..=100.0),
        );
        r |= yaw_edit(ui, &mut sky.rotation);
        Some(r)
    });

    typed_section::<EnvironmentLight>(ui, world, registry, history, e, id, "Environment light", |ui, env| {
        property(ui, "Cubemap", |ui| ui.label(format!("#{}", env.cubemap.index())));
        let mut r = drag_row(
            ui,
            "Diffuse",
            egui::DragValue::new(&mut env.diffuse_intensity)
                .speed(0.01)
                .range(0.0..=100.0),
        );
        r |= drag_row(
            ui,
            "Specular",
            egui::DragValue::new(&mut env.specular_intensity)
                .speed(0.01)
                .range(0.0..=100.0),
        );
        r |= yaw_edit(ui, &mut env.rotation);
        Some(r)
    });

    typed_section::<DirectionalShadow>(ui, world, registry, history, e, id, "Directional shadow", |ui, sh| {
        let sizes = [(512u32, "512"), (1024, "1024"), (2048, "2048"), (4096, "4096")];
        let mut r = choice_edit(ui, "Map size", &mut sh.map_size, &sizes);
        r |= choice_edit(ui, "Filter", &mut sh.filter, &SHADOW_FILTERS);
        r |= property(ui, "Strength", |ui| {
            fill_slider(ui, egui::Slider::new(&mut sh.strength, 0.0..=1.0))
        });
        r |= drag_row(
            ui,
            "Depth bias",
            egui::DragValue::new(&mut sh.depth_bias).speed(0.001).range(0.0..=10.0),
        );
        r |= drag_row(
            ui,
            "Normal bias",
            egui::DragValue::new(&mut sh.normal_bias).speed(0.01).range(0.0..=10.0),
        );
        r |= drag_row(
            ui,
            "Slope bias",
            egui::DragValue::new(&mut sh.slope_bias).speed(0.01).range(0.0..=10.0),
        );
        r |= drag_row(
            ui,
            "Caster depth",
            egui::DragValue::new(&mut sh.caster_depth).speed(1.0).range(0.0..=1e5),
        );
        Some(r)
    });

    typed_section::<Visibility>(ui, world, registry, history, e, id, "Visibility", |ui, v| {
        let mut visible = *v == Visibility::Visible;
        let r = property(ui, "Visible", |ui| ui.checkbox(&mut visible, ""));
        *v = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        Some(r)
    });

    let mesh_changed = typed_section::<MeshSource>(ui, world, registry, history, e, id, "Mesh source", |ui, s| {
        let kind = s.label();
        let r = match s {
            MeshSource::Cube { size } | MeshSource::Plane { size } => property(ui, "Size", |ui| {
                fill_drag(ui, egui::DragValue::new(size).speed(0.01).range(0.001..=1e4))
            }),
            MeshSource::Sphere {
                radius,
                sectors,
                stacks,
            } => {
                let r = property(ui, "Radius", |ui| {
                    fill_drag(ui, egui::DragValue::new(radius).speed(0.01).range(0.001..=1e4))
                });
                r | property(ui, "Segments", |ui| {
                    pair(
                        ui,
                        egui::DragValue::new(sectors).range(3..=256).suffix(" sec"),
                        egui::DragValue::new(stacks).range(2..=256).suffix(" stk"),
                        4,
                    )
                })
            }
            MeshSource::Data(m) => property(ui, "Data", |ui| {
                ui.label(format!("{} verts · {} tris", m.vertex_count(), m.triangle_count()))
            }),
        };
        property(ui, "Kind", |ui| ui.label(kind));
        Some(r)
    });

    if let Ok(mesh) = world.get::<&Mesh3d>(e).map(|m| *m) {
        section(
            ui,
            ("space3d_inspector", "Mesh3d"),
            "Mesh",
            false,
            |_| {},
            |ui| {
                property(ui, "Handle", |ui| ui.label(format!("#{}", mesh.0.index())));
                if let Some(a) = assets {
                    if let Some(t) = a.mesh_triangles(mesh.0) {
                        property(ui, "Triangles", |ui| ui.label(t.to_string()));
                    }
                    if let Some(b) = a.mesh_aabb(mesh.0) {
                        property(ui, "Bounds", |ui| ui.label(format!("{:.2} .. {:.2}", b.min, b.max)));
                    }
                }
            },
        );
    }

    json_sections(ui, world, registry, history, state, e, id);
    add_component_ui(ui, world, registry, history, state, e, id);
    mesh_changed
}

fn json_sections(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &ComponentRegistry,
    history: &mut EditorHistory,
    state: &mut InspectorState,
    e: Entity,
    id: StableId,
) {
    let values: BTreeMap<String, ComponentValue> = component_values(world, registry, e).unwrap_or_default();
    // Components being added have no value yet but an open editor.
    let mut names_shown: Vec<String> = values
        .keys()
        .filter(|n| !TYPED.contains(&n.as_str()))
        .cloned()
        .collect();
    for n in state.json_edits.keys() {
        if !names_shown.contains(n) && !values.contains_key(n) {
            names_shown.push(n.clone());
        }
    }
    for name in names_shown {
        let current = values.get(&name).cloned();
        let editing = state.json_edits.contains_key(&name);
        let (actions, _) = section(
            ui,
            ("space3d_inspector_json", &name),
            &name,
            true,
            |ui| {
                let remove =
                    can_remove(&name) && current.is_some() && icon_button(ui, "×", &format!("Remove {name}")).clicked();
                let edit = !editing && current.is_some() && ui.button("Edit JSON").clicked();
                (remove, edit)
            },
            |ui| {
                if let Some(text) = state.json_edits.get_mut(&name) {
                    ui.add(
                        egui::TextEdit::multiline(text)
                            .code_editor()
                            .desired_rows(4)
                            .desired_width(f32::INFINITY),
                    );
                    let (mut apply, mut cancel) = (false, false);
                    ui.horizontal(|ui| {
                        apply = ui.button("Apply").clicked();
                        cancel = ui.button("Cancel").clicked();
                    });
                    if apply {
                        let result = serde_json::from_str::<ComponentValue>(text)
                            .map_err(|err| err.to_string())
                            .and_then(|v| set_component_value(world, registry, e, &name, v.clone()).map(|_| v));
                        match result {
                            Ok(v) => {
                                history.push(EditCommand::set_json(
                                    format!("Edit {name}"),
                                    id,
                                    name.clone(),
                                    current.clone(),
                                    Some(v),
                                ));
                                state.json_edits.remove(&name);
                                state.json_errors.remove(&name);
                            }
                            Err(err) => {
                                state.json_errors.insert(name.clone(), err);
                            }
                        }
                    }
                    if cancel {
                        state.json_edits.remove(&name);
                        state.json_errors.remove(&name);
                    }
                    if let Some(err) = state.json_errors.get(&name) {
                        ui.colored_label(ui.visuals().error_fg_color, err);
                    }
                } else if let Some(v) = &current {
                    let compact = serde_json::to_string(v).unwrap_or_default();
                    ui.add(egui::Label::new(egui::RichText::new(&compact).monospace()).wrap());
                }
            },
        );
        let (remove, edit) = actions;
        if edit && let Some(v) = &current {
            let pretty = serde_json::to_string_pretty(v).unwrap_or_default();
            state.json_edits.insert(name.clone(), pretty);
        }
        if remove {
            remove_component(world, e, &name);
            history.push(EditCommand::set_json(
                format!("Remove {name}"),
                id,
                name.clone(),
                current.clone(),
                None,
            ));
        }
    }
}

/// Default value for a known component name.
fn insert_default(world: &mut World, e: Entity, name: &str) -> bool {
    let r = match name {
        "Name" => world.insert_one(e, Name("Entity".into())),
        space3d_scene::TRANSFORM => world.insert_one(e, Transform::IDENTITY),
        names::MATERIAL => world.insert_one(e, Material::default()),
        names::DIRECTIONAL_LIGHT => world.insert_one(e, DirectionalLight::default()),
        names::POINT_LIGHT => world.insert_one(e, PointLight::default()),
        names::CAMERA => world.insert_one(e, Camera3d::default()),
        names::VISIBILITY => world.insert_one(e, Visibility::Visible),
        names::MESH_SOURCE => world.insert_one(e, MeshSource::Cube { size: 1.0 }),
        names::CAMERA_FRAME => world.insert_one(e, CameraFrame::default()),
        names::POST_PROCESS => world.insert_one(e, PostProcess::default()),
        names::DIRECTIONAL_SHADOW => world.insert_one(e, DirectionalShadow::default()),
        _ => return false,
    };
    r.is_ok()
}

fn add_component_ui(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &ComponentRegistry,
    history: &mut EditorHistory,
    state: &mut InspectorState,
    e: Entity,
    id: StableId,
) {
    let present = component_values(world, registry, e).unwrap_or_default();
    let options: Vec<String> = registry
        .names()
        .filter(|n| !HIDDEN.contains(n) && !present.contains_key(*n) && *n != space3d_scene::PARENT)
        .map(str::to_owned)
        .collect();
    if options.is_empty() {
        return;
    }
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        // "Add" at the right end, the combo box fills the rest of the row.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let add = ui.add_enabled(!state.add_pick.is_empty(), egui::Button::new("Add"));
            egui::ComboBox::from_id_salt("space3d_add_component")
                .width(ui.available_width())
                .selected_text(if state.add_pick.is_empty() {
                    "Add component..."
                } else {
                    &state.add_pick
                })
                .show_ui(ui, |ui| {
                    for o in &options {
                        ui.selectable_value(&mut state.add_pick, o.clone(), o);
                    }
                });
            if add.clicked() {
                let name = std::mem::take(&mut state.add_pick);
                if insert_default(world, e, &name) {
                    let after = component_values(world, registry, e).and_then(|v| v.get(&name).cloned());
                    history.push(EditCommand::set_json(format!("Add {name}"), id, name, None, after));
                } else {
                    // Unknown type: let the user write its JSON value.
                    state.json_edits.insert(name, "{}".into());
                }
            }
        });
    });
}

/// Two drag values sharing the rest of the line in equal cells (`affix`: characters of the
/// longer prefix / suffix).
fn pair(ui: &mut egui::Ui, a: egui::DragValue<'_>, b: egui::DragValue<'_>, affix: usize) -> egui::Response {
    let width = ((ui.available_width() - ui.spacing().item_spacing.x) / 2.0).floor();
    number_field(ui, width, a, affix, None) | number_field(ui, width, b, affix, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_for_known_components() {
        let mut w = World::new();
        let e = w.spawn(());
        for n in TYPED {
            if n != space3d_scene::PARENT {
                assert!(insert_default(&mut w, e, n), "{n}");
            }
        }
        assert!(!insert_default(&mut w, e, "Custom"));
        assert!(w.satisfies::<(&Camera3d, &MeshSource, &Visibility)>(e));
        assert!(w.satisfies::<(&CameraFrame, &PostProcess, &DirectionalShadow)>(e));
    }
}
