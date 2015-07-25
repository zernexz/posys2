#ifndef __SERIALIZE_HPP_INCLUDED__
#define __SERIALIZE_HPP_INCLUDED__

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
 



BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
    namespace serialization {

        /** Serialization support for cv::Mat */
        template<class Archive>
        void save(Archive &ar, const cv::Mat &m, const unsigned int __attribute__((unused)) version)
        {
            size_t elem_size = m.elemSize();
            size_t elem_type = m.type();

            ar & m.cols;
            ar & m.rows;
            ar & elem_size;
            ar & elem_type;

            const size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }

        /** Serialization support for cv::Mat */
        template<class Archive>
        void load(Archive &ar, cv::Mat &m, const unsigned int __attribute__((unused)) version)
        {
            int    cols, rows;
            size_t elem_size, elem_type;

            ar & cols;
            ar & rows;
            ar & elem_size;
            ar & elem_type;

            m.create(rows, cols, elem_type);

            size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }
    }
}




BOOST_SERIALIZATION_SPLIT_FREE(::cv::Point2f)
namespace boost {
  namespace serialization {
 
    /** Serialization support for cv::Point2f */
    template<class Archive>
    void save(Archive & ar, const ::cv::Point2f& p, const unsigned int version)
    {
      ar & p.x;
      ar & p.y;
    }
 
    /** Serialization support for cv::Point2f */
    template<class Archive>
    void load(Archive & ar, ::cv::Point2f& p, const unsigned int version)
    {
      ar & p.x;
      ar & p.y;
    }
 
  }
}


BOOST_SERIALIZATION_SPLIT_FREE(::cv::Point3f)
namespace boost {
  namespace serialization {
 
    /** Serialization support for cv::Point3f */
    template<class Archive>
    void save(Archive & ar, const ::cv::Point3f& p, const unsigned int version)
    {
      ar & p.x;
      ar & p.y;
      ar & p.z;
    }
 
    /** Serialization support for cv::Point3f */
    template<class Archive>
    void load(Archive & ar, ::cv::Point3f& p, const unsigned int version)
    {
      ar & p.x;
      ar & p.y;
      ar & p.z;
    }
 
  }
}

#endif
