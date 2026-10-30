use std::collections::HashMap;

use anyhow::Error;

use crate::SIUnitDimensions;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamID(pub usize);

pub struct Parameter<T> {
    value: T,
    si_unit: SIUnitDimensions,
}

pub struct ParameterRegistry {
    params_f: HashMap<ParamID, Parameter<f32>>,
    params_u: HashMap<ParamID, Parameter<u32>>,
    param_id_counter: usize,
}

impl ParameterRegistry {
    pub fn new() -> Self {
        Self {
            params_f: HashMap::new(),
            params_u: HashMap::new(),
            param_id_counter: 0,
        }
    }

    pub fn register_f32(&mut self, value: f32, units: SIUnitDimensions) -> ParamID {
        let id = ParamID(self.param_id_counter);
        self.params_f.insert(
            id, Parameter { value, si_unit: units }
        );
        self.param_id_counter += 1;
        tracing::info!("Registered f32 parameter with id {:?} and dimensions {}", id, units);
        id
    }

    pub fn register_u32(&mut self, value: u32, units: SIUnitDimensions) -> ParamID {
        let id = ParamID(self.param_id_counter);
        self.params_u.insert(
            id, Parameter { value, si_unit: units }
        );
        self.param_id_counter += 1;
        tracing::info!("Registered u32 parameter with id {:?} and dimensions {}", id, units);
        id
    }

    pub fn get_param_f32_units(&self, id: ParamID) -> anyhow::Result<SIUnitDimensions> {
        self.params_f.get(&id).map(|param| param.si_unit).ok_or(anyhow::Error::msg(format!("No f32 parameter with id {:?} found", id)))
    }

    pub fn get_param_u32_units(&self, id: ParamID) -> anyhow::Result<SIUnitDimensions> {
        self.params_u.get(&id).map(|param| param.si_unit).ok_or(anyhow::Error::msg(format!("No u32 parameter with id {:?} found", id)))
    }

    pub fn get_param_u32(&self, id: ParamID) -> anyhow::Result<u32> {
        if let Some(value) = self.params_u.get(&id).map(|param| param.value) {
            return Ok(value);
        } else {
            if let Some(_) = self.params_f.get(&id) {
                return Err(Error::msg(format!("Type mismatch: parameter with id {:?} has type f32, not u32!", id)));
            } else {
                return Err(Error::msg(format!("No u32 parameter with id {:?} found.", id)));
            }
        }
    }

    pub fn get_param_f32(&self, id: ParamID) -> anyhow::Result<f32> {
        if let Some(value) = self.params_f.get(&id).map(|param| param.value) {
            return Ok(value);
        } else {
            if let Some(_) = self.params_u.get(&id) {
                return Err(Error::msg(format!("Type mismatch: parameter with id {:?} has type u32, not f32!", id)));
            } else {
                return Err(Error::msg(format!("No f32 parameter with id {:?} found.", id)));
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::*;

    #[fixture]
    fn empty_registry() -> ParameterRegistry {
        ParameterRegistry::new()
    }

    #[rstest]
    fn test_ids_not_equal(mut empty_registry: ParameterRegistry) {
        let some_param = empty_registry.register_u32(1, SIUnitDimensions::NO_UNIT);
        let some_other_param = empty_registry.register_f32(1000.0, SIUnitDimensions::NO_UNIT);

        assert_ne!(some_param, some_other_param);
    }

    #[rstest]
    fn test_si_units_correct(mut empty_registry: ParameterRegistry) -> anyhow::Result<()> {
        let some_param = empty_registry.register_u32(1, SIUnitDimensions::VELOCITY);
        let some_other_param = empty_registry.register_f32(1000.0, SIUnitDimensions::LENGTH);

        let first_unit = empty_registry.get_param_u32_units(some_param)?;
        let second_unit = empty_registry.get_param_f32_units(some_other_param)?;

        assert_eq!(first_unit, SIUnitDimensions::VELOCITY);
        assert_eq!(second_unit, SIUnitDimensions::LENGTH);
        Ok(())
    }

    #[rstest]
    fn test_id_type_mismatch(mut empty_registry: ParameterRegistry) -> anyhow::Result<()> {
        let some_param = empty_registry.register_u32(1, SIUnitDimensions::VELOCITY);

        match empty_registry.get_param_f32(some_param) {
            Err(_msg) => return Ok(()),
            Ok(_) => return Err(Error::msg("Get param should panic when accessing parameter with wrong type.")),
        }
    }

}