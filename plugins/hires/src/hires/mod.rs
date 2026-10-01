use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct Hires(ObjectSubclass<imp::Hires>)
        @extends gst_video::VideoFilter, gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(Some(plugin), "hires", gst::Rank::NONE, Hires::static_type())
}
