#include "tests.hpp"
#include "../tube-segmentation.cpp"
#include "../parameters.hpp"
#include "../SIPL/Exceptions.hpp"

TEST(TubeSegmentation, WrongFilenameException) {
	paramList parameters = initParameters();
	ASSERT_THROW(run("somefilethatdoesntexist.mhd", parameters), SIPL::IOException);
}
