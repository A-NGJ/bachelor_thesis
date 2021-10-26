;; Auto-generated. Do not edit!


(when (boundp 'vrx_gazebo::UsvDrive)
  (if (not (find-package "VRX_GAZEBO"))
    (make-package "VRX_GAZEBO"))
  (shadow 'UsvDrive (find-package "VRX_GAZEBO")))
(unless (find-package "VRX_GAZEBO::USVDRIVE")
  (make-package "VRX_GAZEBO::USVDRIVE"))

(in-package "ROS")
;;//! \htmlinclude UsvDrive.msg.html


(defclass vrx_gazebo::UsvDrive
  :super ros::object
  :slots (_left _right ))

(defmethod vrx_gazebo::UsvDrive
  (:init
   (&key
    ((:left __left) 0.0)
    ((:right __right) 0.0)
    )
   (send-super :init)
   (setq _left (float __left))
   (setq _right (float __right))
   self)
  (:left
   (&optional __left)
   (if __left (setq _left __left)) _left)
  (:right
   (&optional __right)
   (if __right (setq _right __right)) _right)
  (:serialization-length
   ()
   (+
    ;; float32 _left
    4
    ;; float32 _right
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _left
       (sys::poke _left (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _right
       (sys::poke _right (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _left
     (setq _left (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _right
     (setq _right (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get vrx_gazebo::UsvDrive :md5sum-) "3a927990ab5d5c3d628e2d52b8533e52")
(setf (get vrx_gazebo::UsvDrive :datatype-) "vrx_gazebo/UsvDrive")
(setf (get vrx_gazebo::UsvDrive :definition-)
      "# Standard USV thrust massage - port and starbard

# Thrust command - typically ranges from {-1.0 - 1.0}
float32 left
float32 right
")



(provide :vrx_gazebo/UsvDrive "3a927990ab5d5c3d628e2d52b8533e52")


